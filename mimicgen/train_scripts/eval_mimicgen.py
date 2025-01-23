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
from mimicgen.train_scripts.train_debug import load_init_states

def sensor_customize_test_tiago_cup(env):

    env.policy_rollout = True

    # set up sensor positions
    sensor = env.env._external_sensors['external_sensor0']
    # facing robot
    sensor.set_position_orientation(
        position=torch.tensor([ 1.0304, -0.0309,  1.0272]),
        orientation=torch.tensor([0.2690, 0.2659, 0.6509, 0.6583]),
        )
    sensor.image_height = 180
    sensor.image_width = 320
    sensor._add_modality_to_backend(modality='depth_linear')
    sensor._modalities = {"depth_linear", "rgb"}

    # load basic metadata from training file
    print("\n==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # change the density of the objects
    import omnigibson as og
    state = og.sim.dump_state()
    og.sim.stop()

    coffee_cup = env.env.scene.object_registry("name", "coffee_cup")
    coffee_cup.links['base_link'].density = 30

    paper_cup = env.env.scene.object_registry("name", "paper_cup")
    paper_cup.links['base_link'].density = 100

    og.sim.play()
    og.sim.load_state(state)

    # for _ in range(10): 
    #     og.sim.render()
    
    env.reset()
    for _ in range(2): og.sim.step()

    import torch as th
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([ 1.7492, -0.0424,  1.5371]),
        orientation=th.tensor([0.3379, 0.3417, 0.6236, 0.6166]),
    )
    # for _ in range(5): og.sim.render()

    return env

def evaluate_w_rollout(config, mg_config, device, load_checkpoint_folder, check_action_plot=False, start_epoch=1000):
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

    log_dir = os.path.join(load_checkpoint_folder, "logs")
    ckpt_dir = os.path.join(load_checkpoint_folder, "models")
    video_dir = os.path.join(load_checkpoint_folder, "videos")
    time_str = load_checkpoint_folder.split('/')[-1]
    
    # if log_dir and video_dir does not exist, create it -> happen when locally eval runs on cluster
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        
    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # # get the unique id of the run
    # import wandb
    # api = wandb.Api()
    # # project = api.project("tiago_cup")
    # runs = api.runs("tiago_cup")
    # for run in runs:
    #     if run.name == time_str:
    #         run_id = run.id
    # print(run_id)


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
        verbose=False
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
        init_curobo=False,
    )

    env = sensor_customize_test_tiago_cup(env)

    env = EnvUtils.wrap_env_from_config(env, config=config) # apply environment warpper, if applicable
    envs[env.name] = env
    print(envs[env.name])
    
    print("")

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # change the wandb name to the current time_str
    if config.experiment.logging.log_wandb:
        config.unlock()
        config.experiment.name = time_str
        config.lock()


    print("\n============= Loading data =============")
    print("")

    # data_logger = DataLogger(
    #     log_dir,
    #     config,
    #     log_tb=config.experiment.logging.log_tb,
    #     log_wandb=config.experiment.logging.log_wandb,
    #     run_id = run_id,
    # )
    # get stats for normalization observations and actions
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    
    # now the actions are retrieved from the model saved checkpoint
    # # maybe retreve statistics for normalizing observations
    # obs_normalization_stats = None
    # if config.train.hdf5_normalize_obs:
    #     obs_normalization_stats = trainset.get_obs_normalization_stats()

    # # rename mean to offset, std to scale
    # if obs_normalization_stats is not None:
    #     obs_normalization_stats = {
    #         k: {
    #             "offset": v["mean"],
    #             "scale": v["std"],
    #         }
    #         for k, v in obs_normalization_stats.items()
    #     }
    # action_normalization_stats = trainset.get_action_normalization_stats()

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")


    demo_name = trainset.demos[0]
    demo_actions = trainset.get_action_traj(demo_name)['actions']
    replay_from_demo = False
    if replay_from_demo:
        print("\n============= Start replaying the actions from the demostration =============")
        print("")

        env.reset()
        for step in range(demo_actions.shape[0]):
            ob_dict, r, done, truncated, _ = env.step(demo_actions[step])
            print("step: {}, reward: {}, done: {}, truncated: {}".format(step, r, done, truncated))
            if done:
                break
        
        print('finished policy rollout one episode')
        breakpoint()
        # exit the python code
        sys.exit()

    init_states = None

    check_performance_on_trianing_config = True
    if check_performance_on_trianing_config:
        # load initial states in the collected demo, so that we can check the trianing performance
        init_states_all = load_init_states(data_name='D1_10')
        init_states_list = []
        for demo_key in init_states_all:
            init_states = {} 
            # random sample a key from the demo 
            init_states["states"] = init_states_all[demo_key]
            init_states_list.append(init_states)

    print("\n============= Start rollout evaluation =============")
    print("")

    best_valid_loss = None
    # TODO: need to change the return and success rate in the environment
    best_return = {k: -np.inf for k in envs}
    best_success_rate = {k: -1. for k in envs}
    last_ckpt_time = time.time()
    
    # Evaluate the model by by running rollouts

    # do rollouts at fixed rate or if it's time to save a new ckpt
    video_paths = None
    rollout_check = True
    did_rollouts = False

    # read file names in the load_checkpoint_folder
    load_checkpoint_files = os.listdir(ckpt_dir)

    epoch_list = [int(file_name.split('_')[-1].split('.')[0]) for file_name in load_checkpoint_files]
    epoch_list.sort()

    # epoch_list = [epoch_list[-1]]
    epoch_list = [epoch for epoch in epoch_list if epoch >= start_epoch]

    if epoch_list == []:
        print('No checkpoint found for epoch >= {}'.format(start_epoch))
        return None
    
    epoch_list.sort(reverse=True)   

    # epoch_list = [100, 300, 500, 700, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2200]
    epoch_list = [1500, 2200]


    start_time = time.time()
    
    all_epoch_rollout_logs_save_to_ext = {}

    for epoch in epoch_list:
        print("")
        print("\n============= Epoch {} Rollouts =============".format(epoch))
        print("")

        load_checkpoint_path = os.path.join(ckpt_dir, 'model_epoch_{}.pth'.format(epoch))
        if not os.path.exists(load_checkpoint_path):
            print('Checkpoint at provided path {} not found!'.format(load_checkpoint_path))
            # raise Exception("Checkpoint at provided path {} not found!".format(load_checkpoint_path))
            continue

        rollout_policy, ckpt_dict, rollout_model = FileUtils.policy_from_checkpoint(device=device, ckpt_path=load_checkpoint_path, ckpt_dict=None, verbose=False)

        # debugging whether model are different
        check_whether_model_are_different = False
        if check_whether_model_are_different:
            print(f"Loaded model weights from checkpoint")
            breakpoint()
            sc_weight = ckpt_dict['model']['nets']['policy.obs_encoder.nets.obs.obs_nets.combined::point_cloud.layers.0.weight']
            load_checkpoint_path = os.path.join(ckpt_dir, 'model_epoch_{}.pth'.format(10))
            rollout_policy, ckpt_dict_0, rollout_model_0 = FileUtils.policy_from_checkpoint(device=device, ckpt_path=load_checkpoint_path, ckpt_dict=None, verbose=True)
            sc_weight_0 = ckpt_dict_0['model']['nets']['policy.obs_encoder.nets.obs.obs_nets.combined::point_cloud.layers.0.weight']
            print(sc_weight == sc_weight_0)

        num_episodes = config.experiment.rollout.n
        print('start rollouts')
        all_rollout_logs, video_paths, action_info = TrainUtils.rollout_with_stats(
            policy=rollout_policy,
            envs=envs,
            horizon=config.experiment.rollout.horizon,
            use_goals=config.use_goals,
            num_episodes=num_episodes,
            render=False,
            video_dir=video_dir if config.experiment.render_video else None,
            epoch=epoch,
            video_skip=1, #config.experiment.get("video_skip", 5),
            terminate_on_success=config.experiment.rollout.terminate_on_success,
            demo_actions=demo_actions,
            check_action_plot=check_action_plot,
            verbose=True,
            init_states_list=init_states_list,
        )

        # summarize results from rollouts to tensorboard and terminal
        for env_name in all_rollout_logs:
            rollout_logs = all_rollout_logs[env_name]
            for k, v in rollout_logs.items():
                if k.startswith("Time_"):
                    print("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                    # data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
                else:
                    print("Rollout/{}/{}".format(k, env_name), v, epoch)
                    # data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

            print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
            print('Env: {}'.format(env_name))
        
        epoch_ckpt_name = "model_epoch_{}".format(epoch)
        # checkpoint and video saving logic
        updated_stats = TrainUtils.should_save_from_rollout_logs(
            all_rollout_logs=all_rollout_logs,
            best_return=best_return,
            best_success_rate=best_success_rate,
            epoch_ckpt_name=epoch_ckpt_name,
            save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
            save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
        ) # saved based on the best_rollout_success_rate
        # {'best_return': {'test_tiago_cup_D0': 1.0}, 'best_success_rate': {'test_tiago_cup_D0': 1.0}, 'epoch_ckpt_name': 'model_epoch_1860_test_tiago_cup_D0_success_1.0', 'should_save_ckpt': True, 'ckpt_reason': 'success'}

        best_return = updated_stats["best_return"]
        best_success_rate = updated_stats["best_success_rate"]
        epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
        should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
        if updated_stats["ckpt_reason"] is not None:
            ckpt_reason = updated_stats["ckpt_reason"] # 'success'
        did_rollouts = True

        # Only keep saved videos if the ckpt should be saved (but not because of validation score)
        should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
        # config.experiment.keep_all_videos is currently true so the following will not call
        if video_paths is not None and not should_save_video:
            print('breakpoint in should save video')
            breakpoint()
            for env_name in video_paths:
                os.remove(video_paths[env_name])
        print("")
        print('best_return', best_return, 'best_success_rate', best_success_rate, 'epoch_ckpt_name', epoch_ckpt_name)
        print("")

        # Save model checkpoints based on conditions (success rate, validation loss, etc)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=rollout_model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=rollout_policy.obs_normalization_stats,
                action_normalization_stats=rollout_policy.action_normalization_stats,
            )
        all_epoch_rollout_logs_save_to_ext[epoch] = {}
        for env_name in all_rollout_logs:
            rollout_logs = all_rollout_logs[env_name]
            all_epoch_rollout_logs_save_to_ext[epoch][env_name] = rollout_logs

        # save all_epoch_success_rate to a txt file
        with open(os.path.join(load_checkpoint_folder, 'all_epoch_rollout_logs_save_to_ext.txt'), 'w') as f:
            f.write(str(all_epoch_rollout_logs_save_to_ext))
    
        # with open(os.path.join(load_checkpoint_folder, 'all_epoch_rollout_logs_save_to_ext.txt'), "w") as f:
        #     json.dump(str(all_epoch_rollout_logs_save_to_ext), f, indent=4)

    # Finally, log memory usage in MB
    process = psutil.Process(os.getpid())
    mem_usage = int(process.memory_info().rss / 1000000)
    print("System/RAM Usage (MB)", mem_usage, epoch)
    print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))
    print("")
    print('*********************************total time used for rollout evaluation')
    print(time.time() - start_time)
    print("")
    
    
    if env_meta["type"] == EnvUtils.EB.EnvType.OG_TYPE:
        import omnigibson as og
        og.shutdown()

    return None


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
    config.train.cuda = True # TODO: to save memory, put the model on cpu
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    print('device', device)
    # breakpoint()

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    important_stats = None
    try:
        important_stats = evaluate_w_rollout(
            config=config, 
            mg_config=mg_config, 
            device=device,
            load_checkpoint_folder=args.load_checkpoint_folder,
            check_action_plot=args.debug,
            start_epoch=args.start_epoch,
            )
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)

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
        help="set this flag to plot actions debugging purposes"
    )

    parser.add_argument(
        "--load_checkpoint_folder",
        type=str,
        default=None,
        help="the checkpoint folder used to load the model",
    )

    parser.add_argument(
        "--start_epoch",
        type=int,
        default=1000,
        help="the start epoch to evaluate the model",
    )

    # globals()['POLICY_ROLLOUT'] = True

    # breakpoint()
    args = parser.parse_args()
    main(args)

