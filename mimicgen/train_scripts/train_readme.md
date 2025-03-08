## Readme for training

### Data processing
We need to process the generated data to match the robomimic dataset. Each processed demostration contains obs, next_obs, actions, rewards, dones.

If get point cloud information
```
python mimicgen/train_scripts/train_prep_data.py --file_path datasets/core_datasets_og/test_r1_cup/demo_src_test_r1_cup_task_D0/demo.hdf5 --output_path datasets/generated_data/test_r1_cup/robomimic_dataset_D0_ds.hdf5 --split_ratio 0.5 --num_pcd_samples 4096 --fps --vis_sign --with_color
```

If process rgb 
```
python mimicgen/train_scripts/train_prep_data.py --file_path datasets/core_datasets_og/test_r1_cup/demo_src_test_r1_cup_task_D0/demo.hdf5 --output_path datasets/generated_data/test_r1_cup/robomimic_dataset_D0_ds.hdf5 --split_ratio 0.5 --obs_type rgb
```

### Train locally
First need to construct a training json script in ```mimicgen/train_scripts/train_configs/``` folder.

Then you can run the follwing script to train a policy. Make sure to change the ```--config``` to the desired training config and ```--mg_config``` to the desired randomization config. 

```
python mimicgen/train_scripts/train_mimicgen.py --mg_config mimicgen/train_scripts/mg_configs/demo_src_test_r1_cup_task_D0.json --config mimicgen/train_scripts/train_configs/r1_D0_jpos_colorPCD_lr0001_b128_transformer.json
```

The reference bash script is in train_r1_local.sh.


### some depenfencies might be missing

```
open3d
pygpg
fpsample
diffusers
```