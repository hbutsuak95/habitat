import argparse

import os
import shutil


def split_data(split: float = 0.2, path: str = './data/umaze_random') -> None:
    """
    Split dataset into train and validation for a given maze
    Note: To see expected datastructure see dataset.py.
    The following datastructure will be generated:
    /datamodule
        /env_1
            /train
                /traj_0
                    goal.png
                    goal.json
                    /images
                        0.png
                        1.png
                        ...
                    /meta
                        0.json
                        1.json
                        ...
                /traj_1
                ...
            /eval
                /traj_n
                    goal.png
                    goal.json
                    /images
                        0.png
                        1.png
                        ...
                    /meta
                        0.json
                        1.json
                        ...
                /traj_m
                ...
        /env_2
        ...
    Args:
        split: Percentage of trajectories used for validation
        path: Path of the dataset (maze)
    """
    # dst locations
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')

    # Check if train is already created
    if os.path.isdir(train_path):
        if os.listdir(train_path):
            print("Train and Val split already done.")
            return None
        else:
            print("Train directory is empty, creating...")
    else:
        print("Splitting dataset...")

    # List all subfolders
    trajectory_folders = [f for f in os.listdir(path)]
    # Count number of trajectories
    n_traj = len(trajectory_folders)
    # Grab the first chunk of trajectories as training
    train_trajectories = trajectory_folders[:-round(n_traj * split)]
    val_trajectories = trajectory_folders[-round(n_traj * split):]

    # Create directories
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # Move folders to proper directory
    for train_t in train_trajectories:
        shutil.move(os.path.join(path, train_t), train_path)

    print('Train directory created.')
    for val_t in val_trajectories:
        shutil.move(os.path.join(path, val_t), val_path)
    print('Validation directory created')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Split folder with trajectories into train and validation split.')

    parser.add_argument("--split", "-s", type=float, action='store',
                        help="Percentage of trajectories in validation set", default=0.3, required=False)
    parser.add_argument("--path", "-p", type=str, action='store',
                        help="Path to dataset", default='./data/omaze_random', required=False)

    args = parser.parse_args()
    split_data(**vars(args))
