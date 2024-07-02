import torch
import numpy as np

class PSEElabel():
    def __init__(self,file,FPS):
        self.spare = 60 // FPS

        self.current_time = 0
        self.current_time_idx=0
        self.labels_ts=[] # 有标注的时间list,无重复
        self.loadnpy(file)


    def loadnpy(self,file):
        # load label.npy and get label ts
        self.labels = np.load(file)
        self.ts = self.labels['ts'] # 标注的时间，有重复
        self.labels_ts = np.unique(self.ts)
        self.num_label = len(self.labels_ts)  # 有标注的帧数量


    def get_label_at_t(self,t):
        """
        get labels at time t
        :param t: time t
        :return: the label at time t
        """
        if t in self.labels_ts:
            indx = np.where(self.ts == t)
            tlabels = self.labels[indx] # t时刻的labels
            self.current_time = t
            self.current_time_idx = np.where(self.labels_ts==t)[0][0]
            return tlabels
        else:
            print('t not in label ts')




