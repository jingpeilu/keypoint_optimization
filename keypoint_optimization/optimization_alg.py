from functions import prepare_dataset,train_network, evaluate
import multiprocessing
from multiprocessing import Pool
import numpy as np
import pickle
import yaml
import argparse


class Optimization:
    def __init__(self, label_file = 'data', config = "pose_cfg_test.yaml"):

        self.config = config
        self.label_file = label_file
        infile = open(label_file,'rb')
        df = pickle.load(infile)
        infile.close()

        self.n_candiates = df[list(df.keys())[0]].shape[0]
        self.image_num = len(df.keys())
        with open(config) as file:
            config_list = yaml.load(file, Loader=yaml.FullLoader)

        self.n_keypoints = config_list['num_joints']

        # initialization
        self.feature_idx = list(range(self.n_candiates))
        #print(feature_idx)
        self.weights = [1/len(self.feature_idx)]*len(self.feature_idx)
        self.keypoints = dict()
        for i in range(len(self.feature_idx)):
            self.keypoints[self.feature_idx[i]] = self.weights[i]
        # define groups
        self.groups = [list(range(self.n_candiates))]
    


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x/ e_x.sum(axis=0)

    def update_weights(self,S,S_err,keypoints, gamma = 1):
        w_sum = 0
    
        for i in S:        
            w_sum += keypoints[i]
        
        new_w = self.softmax(-np.array(S_err) * gamma)
        for i in range(len(S)):
            idx = S[i]
            keypoints[idx] = w_sum*new_w[i]
        
        return keypoints

    def keypoins_sampling(self, groups, feature_idx, weights,n_per_group):
        S = []
        for group in groups:
            tmp_idx = [feature_idx[i] for i in group] 
            tmp_weights = [weights[i] for i in group] 
            scale = 1/np.sum(tmp_weights)
            tmp_weights = np.array(tmp_weights) * scale
            index = np.random.choice(tmp_idx, size = n_per_group, p = tmp_weights, replace = False)
            S.append(index)
        return S

    def run_iteration(self,n_iteration):

        for i in range(n_iteration):
            S = self.keypoins_sampling(self.groups, self.feature_idx, self.weights, self.n_keypoints)[0].tolist()
            print(S)
            print("Iteration: " + str(i))
            file1 = open("statistic.txt","a") 
            file1.write("Iteration: " + str(i) + "\n")
            file1.write(str(S) + "\n")
            file1.close()

            # prepare dataset
            p = multiprocessing.Process(target=prepare_dataset,args=(self.image_num,S,self.config))
            p.start()
            p.join()

            # train network
            p = multiprocessing.Process(target=train_network,args=(self.config,))
            p.start()
            p.join()

            # run evaluation
            with Pool(processes=4) as pool:
                result = pool.map(evaluate, [S])

            # update weights

            S_err = result[0]

            self.keypoints = self.update_weights(S,S_err,self.keypoints)
            for i in range(len(self.feature_idx)):
                self.weights[i] = self.keypoints[i]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('n_iter', type=int, help='number of iterations')
    args = parser.parse_args()


    kp_optimization = Optimization()
    kp_optimization.run_iteration(args.n_iter)



