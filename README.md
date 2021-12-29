# Pose Estimation for Robot Manipulators via Keypoint Optimization and Sim-to-Real Transfer

Github repo for "Pose Estimation for Robot Manipulators via Keypoint Optmization and Sim-to-Real Transfer"

arXiv: https://arxiv.org/abs/2010.08054

Website and Dataset: https://sites.google.com/ucsd.edu/keypoint-optimization/

## Usage

### Dependencies
Recommend set up the environment using Anaconda.
- [Deeplabcut(2.0+)](https://github.com/DeepLabCut/DeepLabCut)
- Transforms3d

The following packages and software are required for running the robot simulation.
- [PyRep](https://github.com/stepjam/PyRep)
- CoppeliaSim(4.1)

Code is tested on Ubuntu 20.04 with Python 3.8.

### 1. Data Generation

See "keypoint_optimization/data_generation/generate_data_baxter_arm.ipynb" and "generate_data_baxter_ee.ipynb" for example data generation process.

### 2. keypoint Optimization

Run "keypoint_optimization/optimization_alg.py" to optimize the keypoint selection.

### 3. Pose Estimation

See "evaluation/pose_estimation_baxter_arm.ipynb" for example using optimized keypoint to estimate the robot pose.
