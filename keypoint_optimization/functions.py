from pathlib import Path
import os,sys
import numpy as np
import pandas as pd
import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
import platform
import glob
import pickle
import cv2
import yaml
import time
import copy

from functools import lru_cache

from skimage import io
from skimage.transform import resize
import skimage.color
from skimage.util import img_as_ubyte
import scipy.io as sio
import transforms3d.quaternions as quaternions


from deeplabcut.utils import auxiliaryfunctions, conversioncode, auxfun_models, visualization
from deeplabcut.pose_estimation_tensorflow import training
from deeplabcut.pose_estimation_tensorflow.core import predict as ptf_predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
import tensorflow as tf


################### functions for prepare dataset #################

def format_training_data(label_file,train_index, feature_idx, project_path):
    train_data = []
    matlab_data = []
    infile = open(label_file,'rb')
    df = pickle.load(infile)
    infile.close()
    key_list = list(df.keys())


    def to_matlab_cell(array):
        outer = np.array([[None]], dtype=object)
        outer[0, 0] = array.astype('int64')
        return outer
    
    def _read_image_shape_fast(path):
        return cv2.imread(path).shape

    for i in train_index:
        data = dict()
        # get image path
        filename = key_list[i] #string e.g "labeled-data/video_1/frame105_LEFT_half.png"
        data['image'] = filename
        if os.path.exists(filename):
            img_shape = _read_image_shape_fast(os.path.join(project_path, filename))
            try:
                data['size'] = img_shape[2], img_shape[0], img_shape[1]
            except IndexError:
                data['size'] = 1, img_shape[0], img_shape[1]


            # get joint coordinates
            #temp = df.iloc[i].values[1:].reshape(-1, 2).astype(float)
            #joints = np.c_[range(nbodyparts), temp]
            joints = df[filename].astype(int)#joints[~np.isnan(joints).any(axis=1)].astype(int) # nX3 np.array (joint_num,x,y)
            joints = joints[feature_idx]
            joints[:,0] = list(range(len(feature_idx)))


            # Check that points lie within the image
            inside = np.logical_and(np.logical_and(joints[:, 1] < img_shape[1], joints[:, 1] > 0),
                                    np.logical_and(joints[:, 2] < img_shape[0], joints[:, 2] > 0))
            if not all(inside):
                joints = joints[inside]
            if joints.size:  # Exclude images without labels
                data['joints'] = joints
                train_data.append(data)
                matlab_data.append((np.array([data['image']], dtype='U'),
                                    np.array([data['size']]),
                                    to_matlab_cell(data['joints'])))
                
    matlab_data = np.asarray(matlab_data, dtype=[('image', 'O'), ('size', 'O'), ('joints', 'O')])
    return train_data, matlab_data

def prepare_dataset(image_number:int, feature_idx:list, config:str):
    with open(config) as file:
        config_list = yaml.load(file, Loader=yaml.FullLoader)

    project_path = config_list['project_path']
    train_index = list(range(image_number))
    data, MatlabData = format_training_data('data',train_index, feature_idx, project_path)
    datafilename = "train.mat"
    sio.savemat(os.path.join(project_path,datafilename), {'dataset': MatlabData})


############# functions for training ###################


def train_network(poseconfigfile,shuffle=1,trainingsetindex=0,
                  max_snapshots_to_keep=5,displayiters=None,saveiters=None,maxiters=None,
                  allow_growth=False,gputouse=None,autotune=False,keepdeconvweights=True):

    import tensorflow as tf

    # reload logger.
    import importlib
    import logging

    importlib.reload(logging)
    logging.shutdown()

    from deeplabcut.pose_estimation_tensorflow.core.train import train
    from deeplabcut.utils import auxiliaryfunctions

    tf.compat.v1.reset_default_graph()
    start_path = os.getcwd()

    # Read file path for pose_config file. >> pass it on
    #cfg = auxiliaryfunctions.read_config(config)
    #modelfoldername=auxiliaryfunctions.GetModelFolder(cfg["TrainingFraction"][trainingsetindex],shuffle,cfg)
    #poseconfigfile=Path(os.path.join(cfg['project_path'],str(modelfoldername),"train","pose_cfg.yaml"))

    # Set environment variables
    if autotune is not False: #see: https://github.com/tensorflow/tensorflow/issues/13317
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
    if gputouse is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)
    try:
        train(str(poseconfigfile),displayiters,saveiters,maxiters,max_to_keep=max_snapshots_to_keep,keepdeconvweights=keepdeconvweights,allow_growth=allow_growth) #pass on path and file name for pose_cfg.yaml!
    except BaseException as e:
        raise e
    finally:
        os.chdir(str(start_path))
    print("The network is now trained and ready to evaluate. Use the function 'evaluate_network' to evaluate the network.")
    

#################### functions for evaluation #################

#Change this, add this as parameter to predict_single_image
def predict_single_image(image, sess, inputs, outputs, dlc_cfg):
    """
    Returns pose for one single image
    :param image:
    :return:
    """
    # The size here should be the size of the images on which your CNN was trained on
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_as_ubyte(image)
    pose = ptf_predict.getpose(image, dlc_cfg, sess, inputs, outputs)
    return pose


def overwrite_image(image, points_predicted,scores):

    # TODO: Separate this to another function
    height, width = image.shape[:2]

    #Clipping points so that they don't fall outside the image size
    #points_predicted[:,0] = points_predicted[:,0].clip(0, height-1)
    #points_predicted[:,1] = points_predicted[:,1].clip(0, width-1)

    points_predicted = points_predicted.astype(int)

    # Printing as a circle
    for i in range(len(points_predicted)):
        #print(points)
        if scores[i] > 0.0:
            points = points_predicted[i]
            image = cv2.circle(image,tuple(points), 5, (0,0,255), -1)
            
            image = cv2.putText(image, str(i) + " " + str(round(scores[i],2)), tuple(points), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 1, cv2.LINE_AA)  
    return image


def _read_image_shape_fast(path):
    return cv2.imread(path).shape


def get_camera_matrix(x_resolution = 640, y_resolution = 480, fov = np.pi/3):
    if (x_resolution/y_resolution) > 1:
        fov_x = fov
        fov_y = 2 * np.arctan(np.tan(fov/2) / (x_resolution/y_resolution))
    else:
        fov_x = 2 * np.arctan(np.tan(fov/2) / (x_resolution/y_resolution))
        fov_y = fov
    P = np.array([[-(x_resolution/2)/np.tan(0.5*fov_x),0,(x_resolution/2)],
                  [0,-(y_resolution/2)/np.tan(0.5*fov_y),(y_resolution/2)],
                  [0,0,1]])
    return P

def get_gt(filename,feature_idx,df):
    img_shape = _read_image_shape_fast(filename)
    joints = df[filename].astype(int)#joints[~np.isnan(joints).any(axis=1)].astype(int) # nX3 np.array (joint_num,x,y)
    joints = joints[feature_idx]
    joints[:,0] = list(range(len(feature_idx)))

    # Check that points lie within the image
    inside = np.logical_and(np.logical_and(joints[:, 1] < img_shape[1], joints[:, 1] > 0),
                            np.logical_and(joints[:, 2] < img_shape[0], joints[:, 2] > 0))
    if not all(inside):
        joints = joints[inside]
        
    return joints

def overwrite_image_with_gt(image, results, gt):

    # TODO: Separate this to another function

    # Printing as a circle
    for i in range(len(gt)):
        #print(points)
        err = np.linalg.norm(results[i][:2].astype(int) - gt[i][1:])
        err = np.around(err,1)
        score = int(results[i][2]*100)
        image = cv2.circle(image,tuple(results[i][:2].astype(int)), 5, (255,0,0), -1)
        image = cv2.circle(image,tuple(gt[i][1:]), 6, (0,255,0), 2)
        image = cv2.line(image, tuple(results[i][:2].astype(int)), tuple(gt[i][1:]), (255,0,255), 2)
        image = cv2.putText(image, str(score), tuple(results[i][:2].astype(int)), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,0), 1, cv2.LINE_AA)  
    return image

def pose_to_matrix(pose):
    t = pose[:3].reshape(3,1)
    quat_tmp = pose[3:]
    quat = [quat_tmp[3],quat_tmp[0],quat_tmp[1],quat_tmp[2]]
    R = quaternions.quat2mat(quat)
    T = np.vstack((np.hstack((R,t)),[0,0,0,1]))
    return T

def evaluate(feature_idx, weights_path = 'pose_cfg_test.yamlsnapshot-50000', config= "pose_cfg_test.yaml", label_file = "test/data", pose_label_file = "test/pose"):

    gputouse = None
    use_gpu = False
    
    
    # Suppress scientific notation while printing
    np.set_printoptions(suppress=True)

    ##################################################
    # SETUP everything until image prediction
    ##################################################

    if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]  # was potentially set during training

    if gputouse is not None:  # gpu selection
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gputouse)

    tf.compat.v1.reset_default_graph()

    try:
        dlc_cfg = load_config(str(config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle %s and trainFraction %s does not exist."
            % (shuffle, trainFraction)
        )

    dlc_cfg['init_weights'] = weights_path
    print("Running the weights: " + dlc_cfg['init_weights'])

    # Using GPU for prediction
    # Specifying state of model (snapshot / training state)
    sess, inputs, outputs = ptf_predict.setup_pose_prediction(dlc_cfg,allow_growth=True)
        
    from baxter_fk import Baxter_FK
    baxter = Baxter_FK("keypoint_config.json")

    # load ground truth label    
    infile = open(label_file,'rb')
    df = pickle.load(infile)
    infile.close()
    key_list = list(df.keys())

    infile = open(pose_label_file,'rb')
    df_pose = pickle.load(infile)
    infile.close()
    
    error_list = []
    confidence_list = []
    feature_index_list = []
    err_3d_list = []

    try:  
        os.mkdir("evaluate") 
    except OSError as error:  
        print(error)  

    for i in range(len(key_list)):
        img_file = key_list[i]
        gt = get_gt(img_file,feature_idx,df)

        image = cv2.imread(img_file)
        results = predict_single_image(image, sess, inputs, outputs, dlc_cfg)
        results = results[gt[:,0]]
        points_predicted = results[:,:2].astype(int)
        scores = results[:,2]

        #rgb_image = cv2.imread(img_file.replace("edges","images"))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # keypoint 3d position
        joints = df_pose[img_file]['joints']

        points_3d = np.zeros((1,3))
        for f_id in list(np.array(feature_idx)[gt[:,0]]):
            points_3d = np.vstack((points_3d,baxter.get_3d_position_to_bl(f_id,joints).reshape(-1)[:3]))
        points_3d = points_3d[1:,:].astype(np.float32)
        
        # using EPnP to find the cam-to-base transform
        retval,rvec,tvec = cv2.solvePnP(points_3d, points_predicted.astype(float),get_camera_matrix(),distCoeffs = None, flags = 1)
        R,_ = cv2.Rodrigues(rvec)
        quat_tmp = quaternions.mat2quat(R)
        quat = [quat_tmp[1],quat_tmp[2],quat_tmp[3],quat_tmp[0]]
        pose = np.hstack((R,tvec))
        pred_pose = np.vstack((pose,[0,0,0,1]))
        
        # 3d keypoints
        v_cam = pred_pose @ np.concatenate((points_3d,np.ones((points_3d.shape[0],1))), axis = 1).T
        v_cam_dehomogenize = v_cam[:3,:] / v_cam[3,:]

        gt_pose = pose_to_matrix(np.array(df_pose[img_file]['b_T_c']))
        v_cam_gt = gt_pose @ np.concatenate((points_3d,np.ones((points_3d.shape[0],1))), axis = 1).T
        v_cam_dehomogenize_gt = v_cam_gt[:3,:] / v_cam_gt[3,:]

        #img = overwrite_image_with_gt(rgb_image, results, gt) 
        #plt.imsave("evaluate/"+ str(i) + ".png",img)

        for j in range(gt.shape[0]):
            feature_index_list.append(gt[j,0])
            err = np.linalg.norm(gt[j,1:] - points_predicted[j])
            error_list.append(err)
        
            err_3d = np.linalg.norm(v_cam_dehomogenize[:,j] - v_cam_dehomogenize_gt[:,j])
            err_3d_list.append(err_3d)
            confidence_list.append(scores[j])

            
    confidence_list = np.array(confidence_list)
    error_list = np.array(error_list)
    feature_index_list = np.array(feature_index_list)
    err_3d_list = 50 * np.array(err_3d_list)

    # write data in a file. 
    file1 = open("statistic.txt","a") 
    
    file1.write("Average error: " + str(np.mean(error_list)) + " (" + str(np.std(error_list)) + ")" + "\n")
    file1.write("Average 3d error: " + str(np.mean(err_3d_list)) + " (" + str(np.std(err_3d_list)) + ")" + "\n")
    file1.write("Total error: " + str(np.mean(err_3d_list)+ np.mean(error_list)) + "\n")
    err_list = list()
    for i in range(len(feature_idx)):
        idx = np.where(feature_index_list == i)
        avg_err = np.mean(error_list[idx])
        std_err = np.std(error_list[idx])
        avg_err_3d = np.mean(err_3d_list[idx])
        std_err_3d = np.std(err_3d_list[idx])

        avg_confidence = np.mean(confidence_list[idx])
        err_list.append(np.around(avg_err + avg_err_3d,2))
        file1.write("Feature #" + str(i) 
              + " avg_err: " + str(np.around(avg_err,2)) + " (" + str(np.around(std_err,2)) + ")"
              + " avg_err_3d: " + str(np.around(avg_err_3d,2)) + " (" + str(np.around(std_err_3d,2)) + ")"
              + " total err: " + str(np.around(avg_err + avg_err_3d,2))+ "\n")
    file1.write("\n")
    file1.close() #to change file access modes 

    return err_list


    






