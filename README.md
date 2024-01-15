# EvoDrones
Repository for the evolutionary computing research on minidrone controllers.

## Installation of Simulator

```sh
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip install --upgrade pip
pip install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

```
