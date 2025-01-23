#!/usr/bin/env bash

#remote_dir="/viscam/projects/kdm/real-world-data-task1/session_77139c4c"
#local_dir="/home/weiyu/data_drive/kdm/real-world-data-task1/"

# remote_dir="/viscam/projects/kdm/real_world_processed/annotated_segments"
# local_dir="/home/weiyu/data_drive/kdm/real_world"

### generated data transfer
remote_dir="/svl/u/mengdixu/b1k-datagen/mimicgen/datasets"
local_dir="/home/mengdi/b1k_datagen/mimicgen/datasets/generated_data"

# copy the whole dir without the backslash

echo $remote_dir
echo $local_dir

# rsync -aP mengdixu@scdt.stanford.edu:$remote_dir $local_dir
rsync -avz  $local_dir mengdixu@scdt.stanford.edu:$remote_dir