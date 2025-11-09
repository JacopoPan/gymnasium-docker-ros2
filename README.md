# gymnasium-docker-ros2

A Minimal Working Example (MWE) of a Farama Gymnasium environment
that wraps a multi-container ROS2 simulation using `docker-py` and
is trained with `stable-baselines3`.

## Structure

* `/docker`: Contains `Dockerfile`s for the ROS2 nodes.
* `/ros_workspace`: The ROS2 nodes that make up the simulation.
* `/gym_env`: The `RosSimEnv` class (the core wrapper).
* `train.py`: The `stable-baselines3` training script.

## How to Run

1.  **Install Dependencies:**
    * You must have **ROS2 (Humble)** installed and sourced on your host.
    * You must have **Docker** installed and running.

2.  **Install Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Build the Docker Images:**
    ```bash
    ./build_dockers.sh
    ```

4.  **Run the Training:**
    ```bash
    # Make sure you have sourced your ROS2 environment
    source /opt/ros/humble/setup.bash
    
    python train.py
    ```