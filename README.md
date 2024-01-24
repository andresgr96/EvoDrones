# EvoDrones
Repository for the evolutionary computing research on minidrone controllers. Installation instructions:

## Installation of Simulator

```sh
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones/

conda create -n drones python=3.10
conda activate drones

pip install --upgrade pip
pip install -e . # if needed, `sudo apt install build-essential` to install `gcc` and build `pybullet`

```

## Clone this Repository

```sh
cd gym-pybullet-drones/gym_pybullet_drones
git clone https://github.com/andresgr96/EvoDrones.git
cd EvoDrones

pip install -r requirements.txt
pip install -e .
```

## Neat Config File Documentation
https://neat-python.readthedocs.io/en/latest/config_file.html
