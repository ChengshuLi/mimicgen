"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""
# import doppelmaker
# # doppelmaker.import_og_dependencies()

import argparse
import json
import numpy as np
import time
import datetime
import os
import shutil
import psutil
import sys
import socket
import traceback
import random
import imageio
import numpy as np
from copy import deepcopy

from collections import OrderedDict
import sys
from io import StringIO

import torch
from torch.utils.data import DataLoader

import robomimic
from robomimic.utils.file_utils import get_env_metadata_from_dataset
import robomimic.macros as Macros
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
from robomimic.utils.file_utils import load_dict_from_checkpoint
# from doppelmaker.utils.robomimic_utils import get_epochs_trained, doppel_get_exp_dir, get_env_all_objects_initialization_from_dataset
import omnigibson as og
from omnigibson.objects import *

import mimicgen
import robomimic.utils.env_utils as EnvUtils
import mimicgen.utils.file_utils as MG_FileUtils
import mimicgen.utils.robomimic_utils as RobomimicUtils

from mimicgen.configs import MG_TaskSpec
from mimicgen.configs import config_factory as MG_ConfigFactory
from mimicgen.datagen.data_generator import DataGenerator
from mimicgen.env_interfaces.base import make_interface

# import doppelmaker
# doppelmaker.import_og_dependencies()

# logger code from josiah
def doppel_get_exp_dir(config):
    """
    Create experiment directory from config. If an identical experiment directory
    exists and @auto_remove_exp_dir is False (default), the function will prompt 
    the user on whether to remove and replace it, or keep the existing one and
    add a new subdirectory with the new timestamp for the current run.
    
    Returns:
        log_dir (str): path to created log directory (sub-folder in experiment directory)
        output_dir (str): path to created models directory (sub-folder in experiment directory)
            to store model checkpoints
        video_dir (str): path to video directory (sub-folder in experiment directory)
            to store rollout videos
    """
    # timestamp for directory names
    t_now = time.time()
    time_str = datetime.datetime.fromtimestamp(t_now).strftime('%Y%m%d%H%M%S')

    # create directory for where to dump model parameters, tensorboard logs, and videos
    base_output_dir = os.path.expandvars(os.path.expanduser(config.train.output_dir))
    base_output_dir = os.path.join(base_output_dir, config.experiment.name)

    # only make model directory if model saving is enabled
    output_dir = None
    if config.experiment.save.enabled:
        output_dir = os.path.join(base_output_dir, time_str, "models")
        os.makedirs(output_dir)

    # tensorboard directory
    log_dir = os.path.join(base_output_dir, time_str, "logs")
    os.makedirs(log_dir)

    # video directory
    video_dir = os.path.join(base_output_dir, time_str, "videos")
    os.makedirs(video_dir)

    # # establish sync path for syncing important training results back
    # set_absolute_sync_path(
    #     output_dir=config.train.output_dir,
    #     exp_name=config.experiment.name,
    #     time_str=time_str,
    # )

    return log_dir, output_dir, video_dir, time_str

def train(config, mg_config, device, load_checkpoint_path=None, start_epoch_idx=1, auto_remove_exp=False):
    """
    Train a model using the algorithm.
    """

    # time this run
    start_time = time.time()

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # set num workers
    torch.set_num_threads(1)

    if not auto_remove_exp:
        # if we don't auto_remove existing experiment folder
        # call this to create new time stamp folder in the same experiment folder
        log_dir, ckpt_dir, video_dir, time_str = doppel_get_exp_dir(config)
        # save_dir = os.path.join(config.train.output_dir, config.experiment.name)
        # log_dir, ckpt_dir, video_dir = os.getpwd(), os.getcwd
    else:
        log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config, auto_remove_exp_dir=auto_remove_exp)
        
    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations), load randomizers here, changed to CropRandomizer
    ObsUtils.initialize_obs_utils_with_config(config)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")

    # make sure the dataset exists
    # eval_dataset_cfg = config.train.data[0]
    # dataset_path = os.path.expandvars(os.path.expanduser(eval_dataset_cfg["path"]))
    dataset_path = os.path.expandvars(os.path.expanduser(config.train.data))
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))
    
    print("\n============= shape_meta =============")
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        # ds_format=ds_format,
        verbose=True
    )

    print(shape_meta)
    print("")
    
    print("\n============= Loaded Environment Metadata =============")
    # path to source dataset
    source_dataset_path = os.path.expandvars(os.path.expanduser(mg_config.experiment.source.dataset_path))

    # get environment metadata from dataset
    ds_format = config.train.data_format    # TODO: Why BC-RNN config does not have ds_format?
    env_meta = get_env_metadata_from_dataset(dataset_path=source_dataset_path, ds_format=ds_format)

    # env args: cameras to use come from debug camera video to write, or from observation collection
    if config.experiment.rollout.enabled:
        envs = OrderedDict()

        env = RobomimicUtils.create_env(
            env_meta=env_meta,
            env_class=None,
            env_name=mg_config.experiment.task.name,
            robot=mg_config.experiment.task.robot,
            gripper=mg_config.experiment.task.gripper,
            camera_names=mg_config.obs.camera_names,
            camera_height=mg_config.obs.camera_height,
            camera_width=mg_config.obs.camera_width,
            render=config.experiment.render, 
            render_offscreen=False, #config.experiment.render_video,
            use_image_obs=False,
            use_depth_obs=False,
        )


        # load basic metadata from training file
        print("\n==== Using environment with the following metadata ====")
        print(json.dumps(env.serialize(), indent=4))
        print("")

        import omnigibson as og
        state = og.sim.dump_state()
        og.sim.stop()

        coffee_cup = env.env.scene.object_registry("name", "coffee_cup")
        coffee_cup.links['base_link'].density = 30

        paper_cup = env.env.scene.object_registry("name", "paper_cup")
        paper_cup.links['base_link'].density = 100

        og.sim.play()
        og.sim.load_state(state)
        for _ in range(10): 
            og.sim.render()

        # TODO: the following wrapper line will cause error
        # AssertionError: Invalid wrapper type received! Valid options are: dict_keys(['DataWrapper', 'DataCollectionWrapper', 'DataPlaybackWrapper']), got: None
        # env.wrap_env()

        # change viewer camera position
        # set camera postion
        import torch as th
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([ 1.7492, -0.0424,  1.5371]),
            orientation=th.tensor([0.3379, 0.3417, 0.6236, 0.6166]),
        )
        for _ in range(5): og.sim.render()

        env = EnvUtils.wrap_env_from_config(env, config=config) # apply environment warpper, if applicable
        envs[env.name] = env
        print(envs[env.name])
    
    print("")
    
    print("\n============= New Training Run with Config =============")
    print(config)
    print("")

    
    # # env_all_obj_meta = get_env_all_objects_initialization_from_dataset(dataset_path=dataset_path, ds_format="robomimic")   # TODO: does not implement other ds_format
    # env_all_obj_meta = get_env_all_objects_initialization_from_dataset(dataset_path=dataset_path)
    
    # # update env meta if applicable
    # # from robomimic.utils.script_utils import deep_update
    # # deep_update(env_meta, config.experiment.env_meta_update_dict)



    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)
    
    # # Uncomment the following 3 lines if you want to visualize
    # # og.sim.viewer_camera.add_modality("camera")
    # # og.sim.viewer_camera.add_modality("depth_linear")
    # # og.sim.viewer_camera.set_position_orientation([1.325376001942603, -1.9173018358217684, 1.254665264292596], [0.5581211975257528, 2.0704457018914e-07, -1.4148466577544682e-17, 0.8297594403635169])

    
    # setup for a new training run

    # change the wandb name to the current time_str
    if config.experiment.logging.log_wandb:
        config.unlock()
        config.experiment.name = time_str
        config.lock()

    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    # Load checkpoint
    if load_checkpoint_path:
        ckpt_dict = load_dict_from_checkpoint(load_checkpoint_path)
        model.deserialize(ckpt_dict['model'])
        print(f"Loaded model weights from checkpoint")
    
    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")
    
    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # rename mean to offset, std to scale
    if obs_normalization_stats is not None:
        obs_normalization_stats = {
            k: {
                "offset": v["mean"],
                "scale": v["std"],
            }
            for k, v in obs_normalization_stats.items()
        }

    # maybe retreve statistics for normalizing actions
    # action_normalization_stats = trainset.get_action_normalization_stats()
    # import pdb; pdb.set_trace()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # main training loop
    best_valid_loss = None
    # TODO: need to change the return and success rate in the environment
    best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    last_ckpt_time = time.time()

    # need_sync_results = False #(Macros.RESULTS_SYNC_PATH_ABS is not None)
    # if need_sync_results:
    #     # these paths will be updated after each evaluation
    #     best_ckpt_path_synced = None
    #     best_video_path_synced = None
    #     last_ckpt_path_synced = None
    #     last_video_path_synced = None
    #     log_dir_path_synced = os.path.join(Macros.RESULTS_SYNC_PATH_ABS, "logs")

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    # TODO: need to enable this if loading checkpoint
    # # if we load checkpoint, and did not specify starting epoch idx,
    # # infer starting epoch index from checkpoint name
    # if load_checkpoint_path and start_epoch_idx == 1:
    #     ckpt_name = os.path.basename(load_checkpoint_path)  
    #     start_epoch_idx = get_epochs_trained(ckpt_name) + 1

    start_epoch_idx = 1
    
    for epoch in range(start_epoch_idx, config.train.num_epochs + 1): # epoch numbers start at 1
        step_log = TrainUtils.run_epoch(
            model=model,
            data_loader=train_loader,
            epoch=epoch,
            num_steps=train_num_steps,
            obs_normalization_stats=obs_normalization_stats,
        )
        model.on_epoch_end(epoch)

        # setup checkpoint path
        epoch_ckpt_name = "model_epoch_{}".format(epoch)

        # check for recurring checkpoint saving conditions
        should_save_ckpt = False
        if config.experiment.save.enabled:
            time_check = (config.experiment.save.every_n_seconds is not None) and \
                (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
            epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
            epoch_list_check = (epoch in config.experiment.save.epochs)
            should_save_ckpt = (time_check or epoch_check or epoch_list_check)
        ckpt_reason = None
        if should_save_ckpt:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        print("Train Epoch {}".format(epoch))
        print(json.dumps(step_log, sort_keys=True, indent=4))
        for k, v in step_log.items():
            if k.startswith("Time_"):
                data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
            else:
                data_logger.record("Train/{}".format(k), v, epoch)

        # Evaluate the model on validation set
        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps)
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Valid/{}".format(k), v, epoch)

            print("Validation Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))

            # save checkpoint if achieve new best validation loss
            valid_check = "Loss" in step_log
            if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                best_valid_loss = step_log["Loss"]
                if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                    epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                    should_save_ckpt = True
                    ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        # Evaluate the model by by running rollouts

        # do rollouts at fixed rate or if it's time to save a new ckpt
        video_paths = None
        rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
        did_rollouts = False
        
        if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:

            # wrap model as a RolloutPolicy to prepare for rollouts
            rollout_model = RolloutPolicy(
                model,
                obs_normalization_stats=obs_normalization_stats,
                # action_normalization_stats=action_normalization_stats,
            )

            num_episodes = config.experiment.rollout.n
            print('start roll outs')
            all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
                policy=rollout_model,
                envs=envs,
                horizon=config.experiment.rollout.horizon,
                use_goals=config.use_goals,
                num_episodes=num_episodes,
                render=False,
                video_dir=video_dir if config.experiment.render_video else None,
                epoch=epoch,
                video_skip=1, #config.experiment.get("video_skip", 5),
                terminate_on_success=config.experiment.rollout.terminate_on_success,
            )

            # summarize results from rollouts to tensorboard and terminal
            for env_name in all_rollout_logs:
                rollout_logs = all_rollout_logs[env_name]
                for k, v in rollout_logs.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                    else:
                        data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

                print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
                print('Env: {}'.format(env_name))
                print(json.dumps(rollout_logs, sort_keys=True, indent=4))

            # checkpoint and video saving logic
            updated_stats = TrainUtils.should_save_from_rollout_logs(
                all_rollout_logs=all_rollout_logs,
                best_return=best_return,
                best_success_rate=best_success_rate,
                epoch_ckpt_name=epoch_ckpt_name,
                save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
                save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
            )
            best_return = updated_stats["best_return"]
            best_success_rate = updated_stats["best_success_rate"]
            epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
            should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
            if updated_stats["ckpt_reason"] is not None:
                ckpt_reason = updated_stats["ckpt_reason"]
            did_rollouts = True

        # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
        if video_paths is not None and not should_save_video:
            for env_name in video_paths:
                os.remove(video_paths[env_name])

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats
                # action_normalization_stats=action_normalization_stats,
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # terminate logging
    data_logger.close()

    # # sync logs after closing data logger to make sure everything was transferred
    # if need_sync_results:
    #     print("Sync results back to sync path: {}".format(Macros.RESULTS_SYNC_PATH_ABS))
    #     # sync logs dir
    #     if os.path.exists(log_dir_path_synced):
    #         shutil.rmtree(log_dir_path_synced)
    #     shutil.copytree(log_dir, log_dir_path_synced)
    
    # collect important statistics
    important_stats = dict()
    prefix = "Rollout/Success_Rate/"
    exception_prefix = "Rollout/Exception_Rate/"
    for k in data_logger._data:
        if k.startswith(prefix):
            suffix = k[len(prefix):]
            stats = data_logger.get_stats(k)
            important_stats["{}-max".format(suffix)] = stats["max"]
            important_stats["{}-mean".format(suffix)] = stats["mean"]
        elif k.startswith(exception_prefix):
            suffix = k[len(exception_prefix):]
            stats = data_logger.get_stats(k)
            important_stats["{}-exception-rate-max".format(suffix)] = stats["max"]
            important_stats["{}-exception-rate-mean".format(suffix)] = stats["mean"]

    # add in time taken
    important_stats["time spent (hrs)"] = "{:.2f}".format((time.time() - start_time) / 3600.)

    # write stats to disk
    json_file_path = os.path.join(log_dir, "important_stats.json")
    with open(json_file_path, 'w') as f:
        # preserve original key ordering
        json.dump(important_stats, f, sort_keys=False, indent=4)
    
    if env_meta["type"] == EnvUtils.EB.EnvType.OG_TYPE:
        import omnigibson as og
        og.shutdown()

    return important_stats


def main(args):

    # load training config
    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        # with config.values_unlocked():
        with config.unlocked(): # unlock both key and values
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    # load mg_config 
    with open(args.mg_config, "r") as f:
        ext_cfg = json.load(f)
        # config generator from robomimic generates this part of config unused by MimicGen
        if "meta" in ext_cfg:
            del ext_cfg["meta"]
    mg_config = MG_ConfigFactory(ext_cfg["name"], config_type=ext_cfg["type"])
    with mg_config.values_unlocked():
        mg_config.update(ext_cfg)

    if args.dataset is not None:
        config.train.data = [dict(path=args.dataset)]

    if args.name is not None:
        config.experiment.name = args.name

    if args.output is not None:
        config.train.output_dir = args.output

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        Macros.DEBUG = True

        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    important_stats = None
    try:
        important_stats = train(config=config, mg_config=mg_config, device=device, auto_remove_exp=args.auto_remove_exp)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)
    if important_stats is not None:
        important_stats = json.dumps(important_stats, indent=4)
        print("\nRollout Success Rate Stats")
        print(important_stats)

        # maybe sync important stats back
        if Macros.RESULTS_SYNC_PATH_ABS is not None:
            json_file_path = os.path.join(Macros.RESULTS_SYNC_PATH_ABS, "important_stats.json")
            with open(json_file_path, 'w') as f:
                # preserve original key ordering
                json.dump(important_stats, f, sort_keys=False, indent=4)

    # maybe give slack notification
    if Macros.SLACK_TOKEN is not None:
        from robomimic.scripts.give_slack_notification import give_slack_notif
        msg = "Completed the following training run!\nHostname: {}\nExperiment Name: {}\n".format(socket.gethostname(), config.experiment.name)
        msg += "```{}```".format(res_str)
        if important_stats is not None:
            msg += "\nRollout Success Rate Stats"
            msg += "\n```{}```".format(important_stats)
        give_slack_notif(msg)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # External config file that overwrites default config
    parser.add_argument(
        "--mg_config",
        type=str,
        default=None,
        help="path to a mimicgen config json",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # Output path, to override the one in the config
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="(optional) if provided, override the output folder path defined in the config",
    )

    # force delete the experiment folder if it exists
    parser.add_argument(
        "--auto-remove-exp",
        action='store_true',
        help="force delete the experiment folder if it exists"
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    args = parser.parse_args()
    main(args)

