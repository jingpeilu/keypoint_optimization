# Robust Keypoint Detection and Pose Estimation of Robot Manipulators (WIP)

Github repo for "Robust Keypoint Detection and Pose Estimation of Robot Manipulators with Self-Occlusions via Sim-to-Real Transfer"

arXiv: https://arxiv.org/abs/2010.08054

Website and Dataset: https://sites.google.com/ucsd.edu/keypoint-optimization/

## 1. Data Generation

See "data_generation/generate_data_baxter_arm.ipynb" and "data_generation/generate_data_baxter_ee.ipynb" for example data generation process.

## 2. keypoint Optimization

Run "keypoint_optimization/optimization_alg.py" to optimize the keypoint selection.

## 3. Pose Estimation

See "pose_estimation/pose_estimation_baxter_arm.ipynb" for example using optimized keypoint to estimate the robot pose.
