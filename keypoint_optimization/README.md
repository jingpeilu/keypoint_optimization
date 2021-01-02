## keypoint optimization pipeline

### Miodifications might need to make in "pose_cfg_test.yaml"

Path of initial weights:
```
init_weights: /home/arclab/Desktop/vrep_data/init_weights/resnet_v1_50.ckpt
```
Project path:
```
project_path: /home/arclab/Desktop/baxter_data
```
Number of optimal keypoints:
```
num_joints: 6 # e.g. want to find 6 optimized keypoints
```
Also, remember to modify the "all_joints" and "all_joints_names" accordingly. Make sure number of joints should match with the number specified in "num_joints". Naming doesn't matter.
