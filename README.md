# habitat (Data Collection)

1. Installation 


- Install habitat-sim

- Install habitat-lab 


2. Download test-scene assets 


    ```bash
    python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path /path/to/data/

    python -m habitat_sim.utils.datasets_download --uids habitat_example_objects --data-path /path/to/data/

    python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path /path/to/data/
    ```

3. Set PythonPath

    ```bash
    
    export PYTHONPATH=<path to habitat-sim>:$PYTHONPATH
    ```

4. Using Shortest Path Follower

	- Task Specific 
    ```bash
    
    python examples/shortest_path_follower_task.py --config <path to config file for the task> --out_dir <directory to store data> --num_episodes <# episodes to collect>
    ```

    PS: make sure to download the data required for the task you want to use to collect data. 

    - Scene/Environment specific 
    ```bash
    
    python examples/shortest_path_follower_task.py --config <path to config file for the task> --out_dir <directory to store data> --num_episodes <# episodes to collect>
    ```

    
