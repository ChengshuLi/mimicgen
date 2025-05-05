import h5py
import json
import statistics
import numpy as np

# ================ Inspect hdf5 files =================
# # 1. teleoperation collected data
# f1 = h5py.File("/home/arpit/test_projects/OmniGibson/teleop_collected_data/r1_pick_cup.hdf5", "a")

# # 2. after prepare_src_data.py
# f2 = h5py.File("/home/arpit/test_projects/mimicgen/datasets/source_og/r1_pick_cup.hdf5", "r")
# f2 = h5py.File("/home/arpit/test_projects/mimicgen/datasets/source_og/r1_clean_pan.hdf5", "r")

# # 3. generated data from momagen in MimicGen format
# f3 = h5py.File("/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/r1_dishes_away_no_joint_limit/demo_src_r1_dishes_away_task_D0/tmp_failed/date_05_01_2025_time_00_34_11.hdf5", "r")
# f3 = h5py.File("/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/r1_pick_cup/demo_src_r1_pick_cup_task_D0/demo.hdf5")

# # 4. generated data from momagen in Robomimic format
# f4 = h5py.File("/home/arpit/test_projects/mimicgen/datasets/generated_data/test_tiago_single_arm_cup/robomimic_dataset_floor_filtering_fps_4096_color.hdf5", "r")

# breakpoint()
# =======================================================


# # ============ Modify hdf5 file ==============
# f = h5py.File("/home/arpit/test_projects/OmniGibson/teleop_collected_data/r1_pick_cup.hdf5", "a")
# group = f.require_group("mask")  # Create or get the group

# # Define variable-length UTF-8 string data type
# str_dt = h5py.string_dtype(encoding="utf-8")

# data_list = ["demo_14"]
# # Make sure data is a numpy array with correct dtype
# data_array = np.array(data_list, dtype=str_dt)

# # Create dataset
# group.create_dataset("use", data=data_array, dtype=str_dt)
# # ============================================
    

# ============ Obtain stats from data gen ==============

file_path = "/home/arpit/test_projects/mimicgen/datasets/generated_data_mimicgen_format/core_datasets_og/tidy_table_no_soft_vis/demo_src_r1_tidy_table_task_D0/logs/attempt_000042_succ_0_rate_0.0.json"
# file_path = "/home/arpit/test_projects/mimicgen/datasets/eric/r1_dishes_away_combined_important_stats.json"
with open(file_path, 'r') as f:
    data = json.load(f)
breakpoint()

# # -- Obtain visible stats 
vis_percentages = list()
for i, ep_phase_logs in enumerate(data["all_episode_logs"]["phase_logs"]):
    vis_percentage = ep_phase_logs["0"]["num_frames_with_obj_visible"]
    print("vis_percentage: ", vis_percentage)
    vis_percentages.append(vis_percentage)
print("mean vis percentage: ", np.array(vis_percentages).mean())
print("median vis percentage: ", np.median(np.array(vis_percentages)))

# phases_completed = np.array(data["all_episode_logs"]["phases_completed"])
# err_status = np.array(data["all_episode_logs"]["err_status"])
# phase_logs = data["all_episode_logs"]["phase_logs"]
# task_successes = np.array(data["all_episode_logs"]["task_success"])

# # -- Find episodes that have no MP failure but failed at the task
# counter = 0
# for i, task_success in enumerate(task_successes):
#     if not task_success and err_status[i] == "None":
#         print("idx: ", i, "task_success: ", task_success, "err_status: ", err_status[i])
#         counter += 1
# print("counter: ", counter)

# unique_elements, counts = np.unique(phases_completed, return_counts=True)
# frequency = dict(zip(unique_elements, counts))
# indices = np.where(phases_completed == 3)[0]

# counter = 0
# total_counter = 0

# no_retract_err, invalid_query = 0, 0
# for i, phase_log in enumerate(phase_logs):
#     for k in phase_log.keys():
#         total_counter += 1
#         # if len(phase_log[k]["arm_mp_planning_time"].keys()) > 1:
#         #     counter += 1
#         #     print("len: ", len(phase_log[k]["arm_mp_planning_time"].keys()))
#         #     print("err: ", err_status[i])
#         if "0" in phase_log[k]["full_retract_mp_err"].keys():
#             if phase_log[k]["full_retract_mp_err"]["0"] == "None":
#                 no_retract_err += 1
#             if phase_log[k]["full_retract_mp_err"]["0"] == "Invalid Query":
#                 invalid_query += 1

# print("total_counter: ", total_counter)
# print("counter: ", counter)
# print("no_retract_err: ", no_retract_err)
# print("invalid_query: ", invalid_query)

# # -- Obtain episodes that have task failures (no MP failure)
# for idx in range(len(data["all_episode_logs"]["episode_number"])):
#     if not data["all_episode_logs"]["task_success"][idx] and data["all_episode_logs"]["err_status"][idx] == "None":
#         print("idx: ", idx)

# # -- Obtain time taken for all ep
# print("mean time taken for all ep: ", np.median(data["all_episode_logs"]["time_taken"]))

# # -- Obtain time taken for ep with no MP failure
# lis = list()
# for idx in range(len(data["all_episode_logs"]["episode_number"])):
#     if data["all_episode_logs"]["err_status"][idx] == "None":
#         # print("idx: ", idx, data["all_episode_logs"]["err_status"][idx])
#         lis.append(data["all_episode_logs"]["time_taken"][idx])
# print("mean time taken for ep with no MP failure: ", statistics.mean(lis))
# print("median time taken for ep with no MP failure: ", statistics.median(lis))

# print(" ======================== ")
# # Obtain time taken for ep with MP failure
# lis = list()
# for idx in range(len(data["all_episode_logs"]["episode_number"])):
#     if data["all_episode_logs"]["err_status"][idx] != "None":
#         # print("idx: ", idx, data["all_episode_logs"]["err_status"][idx])
#         lis.append(data["all_episode_logs"]["time_taken"][idx])
# print("mean time taken for ep with MP failure: ", statistics.mean(lis))
# print("median time taken for ep with MP failure: ", statistics.median(lis))

# # ==================================================================    


# # =================== Merge hdf5 files ==========================
# import mimicgen.utils.file_utils as MG_FileUtils
# MG_FileUtils.merge_all_hdf5(
#     folder=tmp_dataset_folder_path,
#     new_hdf5_path=new_dataset_path,
#     delete_folder=True,
# )
# if mg_config.experiment.generation.keep_failed:
#     MG_FileUtils.merge_all_hdf5(
#         folder=tmp_dataset_failed_folder_path,
#         new_hdf5_path=new_failed_dataset_path,
#         delete_folder=True,
#     )
# # ===============================================================


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