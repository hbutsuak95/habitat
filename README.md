# habitat (Data Collection)

1. Installation 

    - Install Habitat-sim in a separate conda environment 


    ```bash
    # We require python>=3.7 and cmake>=3.10
    conda create -n habitat python=3.7 cmake=3.14.0
    conda activate habitat
    ```

    ```bash
    conda install habitat-sim withbullet -c conda-forge -c aihabitat 
    ```
    
    or 

    ```bash
    conda install habitat-sim withbullet headless -c conda-forge -c aihabitat 
    ```
    

    - Install Habitat-lab along with habitat_baselines using the forked [repository](https://github.com/hbutsuak95/habitat-lab.git). 


    ```bash
    git clone git@github.com:hbutsuak95/habitat-lab.git
    cd habitat-lab
    pip install -r requirements.txt
    python setup.py develop --all # install habitat and habitat_baselines
    ```


2. Download test-scene assets 


    ```bash
    python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path ./data

    python -m habitat_sim.utils.datasets_download --uids habitat_example_objects --data-path ./data 

    python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path ./data
    ```


3. Using Shortest Path Follower

	- Task Specific 
    ```bash
    
    python scripts/shortest_path_follower_task.py --config ./configs/tasks/pointnav.yaml --out_dir <directory to store data> --num_episodes <# episodes to collect>
    ```

    PS: make sure to download the data required for the task you want to use to collect data. 

    - Scene/Environment Specific
    ```bash
    
    python scripts/shortest_path_follower_scene.py --scene_id ./data/scene_datasets/habitat-test-scenes/skokloster-castle.glb --out_dir <directory to store data> --num_episodes <# episodes to collect> --max_steps <max steps allowed per episode>
    ```

4. Using RL 

    - Task Specific 

    ```bash
    python -u habitat_baselines/run.py --exp-config habitat_baselines/config/pointnav/ppo_pointnav_example.yaml --run-type train
    ```

    - Scene Specific 

    **Work under progress** 


Currently the data collection methods have been only tested on habitat-test-scenes. We can scale it up to Matplotlib and Gibson datasets in the future. 