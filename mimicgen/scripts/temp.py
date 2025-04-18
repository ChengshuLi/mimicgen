import h5py
import json
import statistics
import numpy as np

# # 1. teleoperation collected data
# f1 = h5py.File("/home/arpit/test_projects/OmniGibson/teleop_collected_data/r1_pick_cup.hdf5", "a")

# # 2. after prepare_src_data.py
f2 = h5py.File("/home/arpit/test_projects/mimicgen/datasets/source_og/r1_pick_cup.hdf5", "r")
# f2 = h5py.File("/home/arpit/test_projects/mimicgen/datasets/source_og/r1_tidy_table.hdf5", "r")

# # 3. generated data from momagen in MimicGen format
# f3 = h5py.File("/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/temp2/demo_src_r1_put_away_cup_task_D2/tmp_failed/date_04_15_2025_time_16_52_23.hdf5", "r")
f3 = h5py.File("/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/r1_pick_cup/demo_src_r1_pick_cup_task_D0/demo.hdf5")

# # 4. generated data from momagen in Robomimic format
# f4 = h5py.File("/home/arpit/test_projects/mimicgen/datasets/generated_data/test_tiago_single_arm_cup/robomimic_dataset_floor_filtering_fps_4096_color.hdf5", "r")

breakpoint()

# # # ============ Modify hdf5 file ==============
# f = h5py.File("/home/arpit/test_projects/OmniGibson/teleop_collected_data/r1_pick_cup.hdf5", "a")
# group = f.require_group("mask")  # Create or get the group

# # Define variable-length UTF-8 string data type
# str_dt = h5py.string_dtype(encoding="utf-8")

# data_list = ["demo_1"]
# # Make sure data is a numpy array with correct dtype
# data_array = np.array(data_list, dtype=str_dt)

# # Create dataset
# group.create_dataset("use", data=data_array, dtype=str_dt)
# # # ============================================
    

# file_path = "/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/temp/demo_src_r1_put_away_cup_task_D2/important_stats.json"
# with open(file_path, 'r') as f:
#     data = json.load(f)
# breakpoint()

# # Obtain episodes that have task failures (no MP failure)
# for idx in range(len(data["all_episode_logs"]["episode_number"])):
#     if not data["all_episode_logs"]["task_success"][idx] and data["all_episode_logs"]["err_status"][idx] == "None":
#         print("idx: ", idx)

# # Obtain time taken for ep with no MP failure
# lis = list()
# for idx in range(len(data["all_episode_logs"]["episode_number"])):
#     if data["all_episode_logs"]["err_status"][idx] == "None":
#         print("idx: ", idx, data["all_episode_logs"]["err_status"][idx])
#         lis.append(data["all_episode_logs"]["time_taken"][idx])
# print("mean time taken for ep with no MP failure: ", statistics.mean(lis))
# print("median time taken for ep with no MP failure: ", statistics.median(lis))

# print(" ======================== ")
# # Obtain time taken for ep with MP failure
# lis = list()
# for idx in range(len(data["all_episode_logs"]["episode_number"])):
#     if data["all_episode_logs"]["err_status"][idx] != "None":
#         print("idx: ", idx, data["all_episode_logs"]["err_status"][idx])
#         lis.append(data["all_episode_logs"]["time_taken"][idx])
# print("mean time taken for ep with MP failure: ", statistics.mean(lis))
# print("median time taken for ep with MP failure: ", statistics.median(lis))


# Analyzing generated data stats


# combine multiple videos into one

# import os
# from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array

# # Folder containing MP4 files
# folder_path = "/home/arpit/test_projects/mimicgen/debug_videos/r1_no_visibility_constraint"

# # Get all MP4 files from the folder and sort them
# video_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".mp4")])

# # Load video clips
# clips = [VideoFileClip(video) for video in video_files]

# # Concatenate videos
# final_video = concatenate_videoclips(clips)

# # Save the merged video
# final_video.write_videofile("merged_video.mp4", codec="libx264", fps=24)

# # Close clips
# for clip in clips:
#     clip.close()



# # Load the two video clips
# video1 = VideoFileClip("/home/arpit/test_projects/mimicgen/debug_videos/r1_no_visibility_constraint/merged_video.mp4")
# video2 = VideoFileClip("/home/arpit/test_projects/mimicgen/debug_videos/r1_with_visibility_constraint/merged_video.mp4")

# # Ensure both videos have the same width
# if video1.w != video2.w:
#     target_width = min(video1.w, video2.w)
#     video1 = video1.resize(width=target_width)
#     video2 = video2.resize(width=target_width)

# # Stack videos vertically
# final_video = clips_array([[video1], [video2]])

# # Save the merged video
# final_video.write_videofile("merged_vertical.mp4", codec="libx264", fps=24)

# # Close clips
# video1.close()
# video2.close()