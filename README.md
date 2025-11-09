# gymnasium-docker-ros2

```sh
git clone https://github.com/JacopoPan/gymnasium-docker-ros2.git
cd gymnasium-docker-ros2/

# Install Anaconda: https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html
conda create -n gdr2 python=3.13 # https://devguide.python.org/versions/
conda activate gdr2

pip3 install --upgrade pip
pip3 install -e .

python3 train.py
```