#!/usr/bin/env bash

#remote_dir="/viscam/projects/kdm/real-world-data-task1/session_77139c4c"
#local_dir="/home/weiyu/data_drive/kdm/real-world-data-task1/"

# remote_dir="/viscam/projects/kdm/real_world_processed/annotated_segments"
# local_dir="/home/weiyu/data_drive/kdm/real_world"

### generated data transfer

# LOG_INDEX=20250122132200
# LOG_INDEX=20250122101119

# LOG_INDEX=20250201214318
# LOG_INDEX=20250201213350
# LOG_INDEX=20250216220938

# for batch 256
LOG_INDEX=20250303001540
LOG_INDEX=20250303004126
LOG_INDEX=20250303004117

# for batch 128
LOG_INDEX=20250303003626
# LOG_INDEX=20250303002402
# LOG_INDEX=20250302235524

for LOG_INDEX in 20250303002402;
do 
    # mkdir -p /home/mengdi/b1k_datagen/mimicgen/logs/test_r1_cup/$LOG_INDEX
    # mkdir -p /home/mengdi/b1k_datagen/mimicgen/logs/test_r1_cup/$LOG_INDEX/models
    # mkdir -p /home/mengdi/b1k_datagen/mimicgen/logs/test_r1_cup/$LOG_INDEX/videos
    # mkdir -p /home/mengdi/b1k_datagen/mimicgen/logs/test_r1_cup/$LOG_INDEX/logs

    remote_dir="/svl/u/mengdixu/b1k-datagen/mimicgen/logs/test_r1_cup/$LOG_INDEX/models/model_epoch_2400.pth"
    local_dir="/home/mengdi/b1k_datagen/mimicgen/logs/test_r1_cup/$LOG_INDEX/models/"
    echo $remote_dir
    echo $local_dir
    rsync -aP mengdixu@scdt.stanford.edu:$remote_dir $local_dir
    
    # remote_dir="/svl/u/mengdixu/b1k-datagen/mimicgen/logs/test_r1_cup/$LOG_INDEX/config.json"
    # local_dir="/home/mengdi/b1k_datagen/mimicgen/logs/test_r1_cup/$LOG_INDEX"
    # echo $remote_dir
    # echo $local_dir
    # rsync -aP mengdixu@scdt.stanford.edu:$remote_dir $local_dir
done

# ssh mengdixu@scdt.stanford.edu "mkdir -p $remote_dir"

# copy the whole dir without the backslash

# echo $remote_dir
# echo $local_dir

# from remote to local
# rsync -avz  $local_dir mengdixu@scdt.stanford.edu:$remote_dir
# rsync -avz --exclude-from='exclude.txt' mengdixu@scdt.stanford.edu:$remote_dir $local_dir

# from remote to local
# rsync -aP mengdixu@scdt.stanford.edu:$remote_dir $local_dir

# # from local to remote
# rsync -aP $local_dir mengdixu@scdt.stanford.edu:$remote_dir 