import h5py
import numpy as np

# f = h5py.File("/tmp/core_datasets_og/test_tiago_single_arm_cup/demo_src_test_tiago_single_arm_cup_task_D1/demo_failed.hdf5", "r")
# f2 = h5py.File("/home/arpit/test_projects/mimicgen/temp_datasets/demo_failed.hdf5", "r")
# f3 = h5py.File("/home/arpit/test_projects/mimicgen/datasets/generated_data/test_tiago_single_arm_cup/robomimic_dataset_floor_filtering_fps_4096_color.hdf5", "r")
# f4 = h5py.File("/home/arpit/test_projects/mimicgen/datasets/source_og/test_tiago_single_arm_cup.hdf5", "r")
# f5 = h5py.File("/home/arpit/test_projects/mimicgen/datasets/source_og/test_r1_cup.hdf5", "r")

# breakpoint()



# combine multiple videos into one

import os
from moviepy.editor import VideoFileClip, concatenate_videoclips, clips_array

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



# Load the two video clips
video1 = VideoFileClip("/home/arpit/test_projects/mimicgen/debug_videos/r1_no_visibility_constraint/merged_video.mp4")
video2 = VideoFileClip("/home/arpit/test_projects/mimicgen/debug_videos/r1_with_visibility_constraint/merged_video.mp4")

# Ensure both videos have the same width
if video1.w != video2.w:
    target_width = min(video1.w, video2.w)
    video1 = video1.resize(width=target_width)
    video2 = video2.resize(width=target_width)

# Stack videos vertically
final_video = clips_array([[video1], [video2]])

# Save the merged video
final_video.write_videofile("merged_vertical.mp4", codec="libx264", fps=24)

# Close clips
video1.close()
video2.close()