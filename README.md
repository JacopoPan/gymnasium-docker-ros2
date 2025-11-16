# gymnasium-docker-ros2

GDR2 shows how to wrap a multi-container Docker/Gazebo/ROS2 app into [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium) and train [`stable-baselines3`](https://github.com/DLR-RM/stable-baselines3)'s PPO, using [`pyzmq`](https://github.com/zeromq/pyzmq) for communication and `gz service` to synchronously step the `/clock`

> [!IMPORTANT]
> This repo is developed using Ubuntu 22.04 with `nvidia-driver-580` on an i9-13 with RTX 3500

```sh
git clone https://github.com/JacopoPan/gymnasium-docker-ros2.git
cd gymnasium-docker-ros2/

# Install Docker Engine: https://docs.docker.com/engine/install/ubuntu/ and https://docs.docker.com/engine/install/linux-postinstall/
docker build -t gdr2-image -f resources/Dockerfile .

# Install Anaconda: https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html
conda create -n gdr2 python=3.13 # https://devguide.python.org/versions/
conda activate gdr2

pip3 install --upgrade pip
pip3 install -e .

python3 test.py --mode step # Manually step the simulation
python3 test.py --mode speed # Check the simulation throughput
python3 test.py --mode learn # Train and test a PPO agent
```

<!--

Test containers with:
docker run -it --rm --env TMUX_OPTS=simulation gdr2-image
docker run -it --rm --env TMUX_OPTS=dynamics gdr2-image

docker exec -it simulation-container tmux attach
docker exec -it dynamics-container tmux attach

docker stop $(docker ps -q) && docker container prune -f && docker network prune -f

>
