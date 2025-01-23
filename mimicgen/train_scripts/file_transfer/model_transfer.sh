#!/usr/bin/env bash

#remote_dir="/viscam/projects/kdm/real-world-data-task1/session_77139c4c"
#local_dir="/home/weiyu/data_drive/kdm/real-world-data-task1/"

# remote_dir="/viscam/projects/kdm/real_world_processed/annotated_segments"
# local_dir="/home/weiyu/data_drive/kdm/real_world"

### generated data transfer

# LOG_INDEX=20250120140017
# LOG_INDEX=20250120135620
# LOG_INDEX=20250120140016
# LOG_INDEX=20250120135621

LOG_INDEX=20250122132200

remote_dir="/svl/u/mengdixu/b1k-datagen/mimicgen/logs/test_tiago_cup/$LOG_INDEX/"
local_dir="/home/mengdi/b1k_datagen/mimicgen/logs/test_tiago_cup/$LOG_INDEX/"

# ssh mengdixu@scdt.stanford.edu "mkdir -p $remote_dir"

# copy the whole dir without the backslash

echo $remote_dir
echo $local_dir

# from remote to local
# rsync -avz  $local_dir mengdixu@scdt.stanford.edu:$remote_dir
# rsync -avz --exclude-from='exclude.txt' mengdixu@scdt.stanford.edu:$remote_dir $local_dir
rsync -aP mengdixu@scdt.stanford.edu:$remote_dir $local_dir
