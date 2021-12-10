import numpy as np
import time
import json 


class Baxter_FK:
    def __init__(self, keypoint_config):

        self.keypoint_config = keypoint_config


    def T_from_DH(self, alp,a,d,the):
        '''
        Transformation matrix fron DH
        '''
        T = np.array([[np.cos(the), -np.sin(the), 0, a],
                    [np.sin(the)*np.cos(alp), np.cos(the)*np.cos(alp), -np.sin(alp), -d*np.sin(alp)],
                    [np.sin(the)*np.sin(alp), np.cos(the)*np.sin(alp), np.cos(alp), d*np.cos(alp)],
                    [0,0,0,1]])
        return T

    def get_bl_T_Jn(self, n, theta):
        '''
        Get joint to base(left) transform using baxter FK
        FK source: https://www.ohio.edu/mechanical-faculty/williams/html/pdf/BaxterKinematics.pdf
        n = 6 for J6 to base
        n = 8 for EE to base
        '''
        assert n in [0,2,4,6,8]
        bl_T_0 = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0.27035],
                        [0,0,0,1]])
        T_7_ee = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0.3683],
                        [0,0,0,1]])
        
        T_0_1 = self.T_from_DH(0,0,0,theta[0])
        T_1_2 = self.T_from_DH(-np.pi/2,0.069,0,theta[1]+np.pi/2)
        T_2_3 = self.T_from_DH(np.pi/2,0,0.36435,theta[2])
        T_3_4 = self.T_from_DH(-np.pi/2,0.069,0,theta[3])
        T_4_5 = self.T_from_DH(np.pi/2,0,0.37429,theta[4])
        T_5_6 = self.T_from_DH(-np.pi/2,0.010,0,theta[5])
        T_6_7 = self.T_from_DH(np.pi/2,0,0,theta[6])
        if n == 0:
            T = T_0_1
        elif n == 2:
            T = bl_T_0  @ T_0_1 @ T_1_2
        elif n == 4:
            T = bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4
        elif n == 6:
            T = bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6
        elif n == 8:
            T = bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6 @ T_6_7 @ T_7_ee
        else:
            raise Exception("Invalid joint number")
        return T

    def get_position_wrt_joints(self, p_id):
        # Opening JSON file 
        with open(self.keypoint_config) as json_file: 
            data = json.load(json_file) 
            
        name = "point_" + str(p_id)
        position = data[name]["position"]
        parent = data[name]["parent"]
        
        return position, parent

    def get_3d_position_to_bl(self, p_id,theta):
        
        position_to_joint, parent_joint = self.get_position_wrt_joints(p_id)
        # list to p_vec
        position_to_joint = np.hstack((position_to_joint,[1])).reshape((4,1))
        if parent_joint == "J0":
            position_to_bl = self.get_bl_T_Jn(0,theta) @ position_to_joint
        elif parent_joint == "J2":
            position_to_bl = self.get_bl_T_Jn(2,theta) @ position_to_joint
        elif parent_joint == "J4":
            position_to_bl = self.get_bl_T_Jn(4,theta) @ position_to_joint
        elif parent_joint == "J6":
            position_to_bl = self.get_bl_T_Jn(6,theta) @ position_to_joint
        elif parent_joint == "EE":
            position_to_bl = self.get_bl_T_Jn(8,theta) @ position_to_joint
        else:
            raise Exception("Sorry, no parent joint found")
        return position_to_bl

     



