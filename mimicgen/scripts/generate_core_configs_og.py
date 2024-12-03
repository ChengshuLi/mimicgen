# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
We utilize robomimic's config generator class to easily generate data generation configs for our
core set of tasks in the paper. It can be modified easily to generate other configs.

The global variables at the top of the file should be configured manually.

See https://robomimic.github.io/docs/tutorials/hyperparam_scan.html for more info.
"""
import os
import json
import shutil

import robomimic
from robomimic.utils.hyperparam_utils import ConfigGenerator

import mimicgen
import mimicgen.utils.config_utils as ConfigUtils
from mimicgen.utils.file_utils import config_generator_to_script_lines


# set path to folder containing src datasets
SRC_DATA_DIR = os.path.join(mimicgen.__path__[0], "../datasets/source_og")

# set base folder for where to copy each base config and generate new config files for data generation
CONFIG_DIR = "/tmp/core_configs_og"

# set base folder for newly generated datasets
OUTPUT_FOLDER = "/tmp/core_datasets_og"

# number of trajectories to generate (or attempt to generate)
NUM_TRAJ = 2

# whether to guarantee that many successful trajectories (e.g. keep running until that many successes, or stop at that many attempts)
GUARANTEE = False

# whether to run a quick debug run instead of full generation
DEBUG = False

# camera settings for collecting observations
CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]
CAMERA_SIZE = (84, 84)

BASE_BASE_CONFIG_PATH = os.path.join(mimicgen.__path__[0], "exps/templates/omnigibson")
BASE_CONFIGS = [
#     os.path.join(BASE_BASE_CONFIG_PATH, "test_pen_book.json"),
#     os.path.join(BASE_BASE_CONFIG_PATH, "test_cabinet.json"),
#     os.path.join(BASE_BASE_CONFIG_PATH, "test_tiago_giftbox.json"),
    # os.path.join(BASE_BASE_CONFIG_PATH, "test_tiago_notebook.json"),
    os.path.join(BASE_BASE_CONFIG_PATH, "test_tiago_cup.json"),
]

def make_generators(base_configs):
    """
    An easy way to make multiple config generators by using different
    settings for each.
    """
    all_settings = [
        # dict(
        #     dataset_path=os.path.join(SRC_DATA_DIR, "test_pen_book.hdf5"),
        #     dataset_name="test_pen_book",
        #     generation_path="{}/test_pen_book".format(OUTPUT_FOLDER),
        #     tasks=["test_pen_book_D0", "test_pen_book_D1"],
        #     task_names=["D0", "D1"],
        #     select_src_per_subtask=False,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     subtask_term_offset_range=[[5, 10], None],
        # ),
        # dict(
        #     dataset_path=os.path.join(SRC_DATA_DIR, "test_cabinet.hdf5"),
        #     dataset_name="test_cabinet",
        #     generation_path="{}/test_cabinet".format(OUTPUT_FOLDER),
        #     tasks=["test_cabinet_D0", "test_cabinet_D1"],
        #     task_names=["D0", "D1"],
        #     select_src_per_subtask=False,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     subtask_term_offset_range=[[5, 10], None],
        # ),
        # dict(
        #     dataset_path=os.path.join(SRC_DATA_DIR, "test_tiago_giftbox.hdf5"),
        #     dataset_name="test_tiago_giftbox",
        #     generation_path="{}/test_tiago_giftbox".format(OUTPUT_FOLDER),
        #     tasks=["test_tiago_giftbox_D0", "test_tiago_giftbox_D1"],
        #     task_names=["D0", "D1"],
        #     select_src_per_subtask=False,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     subtask_term_offset_range=[[5, 10], None],
        # ),
        # dict(
        #     dataset_path=os.path.join(SRC_DATA_DIR, "test_tiago_notebook.hdf5"),
        #     dataset_name="test_tiago_notebook",
        #     generation_path="{}/test_tiago_notebook".format(OUTPUT_FOLDER),
        #     tasks=["test_tiago_notebook_D0", "test_tiago_notebook_D1"],
        #     task_names=["D0", "D1"],
        #     select_src_per_subtask=False,
        #     selection_strategy="random",
        #     selection_strategy_kwargs=None,
        #     subtask_term_offset_range=[[5, 6], [0, 1], None],
        # ),
        dict(
            dataset_path=os.path.join(SRC_DATA_DIR, "test_tiago_cup.hdf5"),
            dataset_name="test_tiago_cup",
            generation_path="{}/test_tiago_cup".format(OUTPUT_FOLDER),
            tasks=["test_tiago_cup_D0", "test_tiago_cup_D1"],
            task_names=["D0", "D1"],
            select_src_per_subtask=False,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            subtask_term_offset_range=[[5, 6], [0, 1], None, [5, 6], [0, 1], None],
        ),
    ]

    assert len(base_configs) == len(all_settings)
    ret = []
    for conf, setting in zip(base_configs, all_settings):
        ret.append(make_generator(os.path.expanduser(conf), setting))
    return ret


def make_generator(config_file, settings):
    """
    Implement this function to setup your own hyperparameter scan.
    Each config generator is created using a base config file (@config_file)
    and a @settings dictionary that can be used to modify which parameters
    are set.
    """
    generator = ConfigGenerator(
        base_config_file=config_file,
        script_file="", # will be overriden in next step
    )

    # set basic settings
    ConfigUtils.set_basic_settings(
        generator=generator,
        group=0,
        source_dataset_path=settings["dataset_path"],
        source_dataset_name=settings["dataset_name"],
        generation_path=settings["generation_path"],
        guarantee=GUARANTEE,
        num_traj=NUM_TRAJ,
        num_src_demos=10,
        max_num_failures=25,
        num_demo_to_render=10,
        num_fail_demo_to_render=25,
        render_video=False,
        verbose=False,
    )

    # set settings for subtasks
    bimanual=True
    if bimanual:
        # now all the configs are from the configuraiton file 
        ConfigUtils.set_subtask_settings_bimanual(
            generator=generator,
            group=0,
            base_config_file=config_file,
            select_src_per_subtask=settings["select_src_per_subtask"],
            # subtask_term_offset_range=settings["subtask_term_offset_range"],
            # selection_strategy=settings.get("selection_strategy", None),
            # selection_strategy_kwargs=settings.get("selection_strategy_kwargs", None),
            # # default settings: action noise 0.05, with 5 interpolation steps
            # # Disable any action noise for now
            # # action_noise=0.05,
            # action_noise=0.0,
            # num_interpolation_steps=5,
            # num_fixed_steps=0,
            verbose=False,
            # arm=False,
        )
    else:
        ConfigUtils.set_subtask_settings(
            generator=generator,
            group=0,
            base_config_file=config_file,
            select_src_per_subtask=settings["select_src_per_subtask"],
            subtask_term_offset_range=settings["subtask_term_offset_range"],
            selection_strategy=settings.get("selection_strategy", None),
            selection_strategy_kwargs=settings.get("selection_strategy_kwargs", None),
            # default settings: action noise 0.05, with 5 interpolation steps
            # Disable any action noise for now
            # action_noise=0.05,
            action_noise=0.0,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            verbose=False,
        )

    # optionally set env interface to use, and type
    # generator.add_param(
    #     key="experiment.task.interface",
    #     name="",
    #     group=0,
    #     values=[settings["task_interface"]],
    # )
    # generator.add_param(
    #     key="experiment.task.interface_type",
    #     name="",
    #     group=0,
    #     values=["robosuite"],
    # )

    # set task to generate data on
    generator.add_param(
        key="experiment.task.name",
        name="task",
        group=1,
        values=settings["tasks"],
        value_names=settings["task_names"],
    )

    # optionally set robot and gripper that will be used for data generation (robosuite-only)
    if settings.get("robots", None) is not None:
        generator.add_param(
            key="experiment.task.robot",
            name="r",
            group=2,
            values=settings["robots"],
        )
    if settings.get("grippers", None) is not None:
        generator.add_param(
            key="experiment.task.gripper",
            name="g",
            group=2,
            values=settings["grippers"],
        )

    # set observation collection settings
    ConfigUtils.set_obs_settings(
        generator=generator,
        group=-1,
        collect_obs=True,
        camera_names=CAMERA_NAMES,
        camera_height=CAMERA_SIZE[0],
        camera_width=CAMERA_SIZE[1],
    )

    if DEBUG:
        # set debug settings
        ConfigUtils.set_debug_settings(
            generator=generator,
            group=-1,
        )

    # seed
    generator.add_param(
        key="experiment.seed",
        name="",
        group=1000000,
        values=[1],
    )

    return generator


def main():

    # make config generators
    generators = make_generators(base_configs=BASE_CONFIGS)

    # maybe remove existing config directory
    config_dir = CONFIG_DIR
    if os.path.exists(config_dir):
        ans = input("Non-empty dir at {} will be removed.\nContinue (y / n)? \n".format(config_dir))
        if ans != "y":
            exit()
        shutil.rmtree(config_dir)

    all_json_files, run_lines = config_generator_to_script_lines(generators, config_dir=config_dir)

    real_run_lines = []
    for line in run_lines:
        line = line.strip().replace("train.py", "generate_dataset.py")
        line += " --auto-remove-exp"
        real_run_lines.append(line)
    run_lines = real_run_lines

    print("configs")
    print(json.dumps(all_json_files, indent=4))
    print("runs")
    print(json.dumps(run_lines, indent=4))


if __name__ == "__main__":
    main()
