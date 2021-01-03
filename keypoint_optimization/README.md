## keypoint optimization pipeline

1. Getting the training and testing images and labels (move 'images' folder, 'test' folder, and 'data' label file from data_generation to this directory)
2. Editing "pose_cfg_test.yaml"
3. Run optimization pipeline to find optimial keypoints
```
python optimization_alg.py [number of iteration]
```

### Miodifications might need to make in "pose_cfg_test.yaml"

Path of initial weights:
```
init_weights: /home/arclab/Desktop/vrep_data/init_weights/resnet_v1_50.ckpt
```
Project path:
```
project_path: /home/arclab/Desktop/baxter_data
```
dataset:
```
dataset: project_path/train.mat
```
Number of optimal keypoints:
```
num_joints: 6 # e.g. want to find 6 optimized keypoints
```
Also, remember to modify the "all_joints" and "all_joints_names" accordingly. Make sure number of joints should match with the number specified in "num_joints". Naming doesn't matter.
