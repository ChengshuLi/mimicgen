import json
import glob
import os

def combine_logs(base_dir, output_file):
    pattern = os.path.join(base_dir, "r1_dishes_away_worker_*/demo_src_r1_dishes_away_task_D0/important_stats.json")
    json_files = sorted(glob.glob(pattern))

    combined_logs = None
    episode_counter = 0

    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)

        logs = data["all_episode_logs"]

        if combined_logs is None:
            combined_logs = {key: [] for key in logs}

        num_episodes = len(logs["episode_number"])
        for i in range(num_episodes):
            for key in logs:
                if key == "episode_number":
                    combined_logs[key].append(episode_counter)
                else:
                    combined_logs[key].append(logs[key][i])
            episode_counter += 1

    with open(output_file, 'w') as f:
        json.dump({"all_episode_logs": combined_logs}, f, indent=2)

# Run it
combine_logs(
    base_dir="/home/arpit/test_projects/mimicgen/datasets/eric/dishes_away_wo_joint_limit",
    output_file="combined_important_stats.json"
)