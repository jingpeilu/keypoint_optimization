from os.path import dirname, join, abspath
import numpy as np
import time
import cv2
from numpy import random

from pathlib import Path
import os,sys
import os.path

from skimage import io
from skimage.transform import resize
import skimage.color
from skimage.util import img_as_ubyte

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.pose_estimation_tensorflow.lib import inferenceutils, trackingutils
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal
import tensorflow as tf


class Keypoint_detector:
    def __init__(self, weights_path, config):

        self.weights_path = weights_path
        self.config = config

        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

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
            self.dlc_cfg = load_config(str(self.config))
        except FileNotFoundError:
            raise FileNotFoundError(
                "It seems the model for shuffle %s and trainFraction %s does not exist."
                % (shuffle, trainFraction)
            )

        self.dlc_cfg['init_weights'] = self.weights_path
        print("Running the weights: " + self.dlc_cfg['init_weights'])


        # Using GPU for prediction
        # Specifying state of model (snapshot / training state)

        self.sess, self.inputs, self.outputs = predict.setup_pose_prediction(self.dlc_cfg,allow_growth=True)


    def predict_single_image(self, image):
        """
        Returns pose for one single image
        :param image:
        :return:
        """
        # The size here should be the size of the images on which your CNN was trained on
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = img_as_ubyte(image)
        pose = predict.getpose(image, self.dlc_cfg, self.sess, self.inputs, self.outputs)
        return pose



    def overwrite_image(self, image, points_predicted,scores):

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
                image = cv2.circle(image,tuple(points), 15, (0,255,0), -1)
                
                #image = cv2.putText(image, str(i) + " " + str(round(scores[i],2)), tuple(points), cv2.FONT_HERSHEY_SIMPLEX,0.2, (0,0,255), 1, cv2.LINE_AA)  
        return image



