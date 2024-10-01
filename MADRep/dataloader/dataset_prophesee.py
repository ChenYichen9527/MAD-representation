import os
import numpy as np
import torch
from dataloader.basedataset import BaseDataLoader
from numpy.lib import recfunctions as rfn


class PropDataset(BaseDataLoader):
    def __init__(self,configs,mode = 'train'):
        super().__init__(configs)

        self.ev_file_list=[]
        self.lb_file_list=[]

        self.get_file_list(mode=mode)
        print(mode+' dataset find: %d samples, %d labels' % (len(self.ev_file_list), len(self.lb_file_list)))

        self.epoch = 0
        self.samples = 0
        self.nr_samples = len(self.lb_file_list)

    def get_file_list(self,mode='train'):
        roots = os.listdir(self.root)
        test_roots=[]
        train_roots = []
        val_roots = []
        for data_root in roots:
            if 'test' in data_root:
                test_roots.append(data_root)
            if 'train' in data_root:
                train_roots.append(data_root)
            if 'val' in data_root:
                val_roots.append(data_root)

        dat_roots=[]
        if mode=='train':
            mode_roots = train_roots

        if mode == 'val':
            mode_roots = val_roots
        if mode=='test':
            mode_roots = test_roots
        if mode=='pre':
            mode_roots = test_roots
            mode='test'

        for mode_roots_ in mode_roots:
            dat_root = os.path.join(self.root, mode_roots_, mode)
            dat_roots.append(dat_root)

        self.ev_file_list = []
        self.lb_file_list = []
        for dat_root_ in dat_roots:
            dat_root_s = os.listdir(dat_root_)
            for root_ in dat_root_s:
                ev_files_root = os.path.join(dat_root_, root_, 'events')
                lb_files_root = os.path.join(dat_root_, root_, 'labels')
                ev_file_list = os.listdir(ev_files_root)
                lb_file_list = os.listdir(lb_files_root)
                ev_file_list.sort(key=lambda x: int(x.split('_ev')[1].split('.')[0]))
                lb_file_list.sort(key=lambda x: int(x.split('_lb')[1].split('.')[0]))
                for ev_file in ev_file_list:
                    path = os.path.join(ev_files_root, ev_file)
                    self.ev_file_list.append(path)
                for lb_file in lb_file_list:
                    path = os.path.join(lb_files_root, lb_file)
                    self.lb_file_list.append(path)

        self.ev_file_list = np.array(self.ev_file_list)
        self.lb_file_list = np.array(self.lb_file_list)


    def __getitem__(self, idx):
        input = {}
        ev_file = self.ev_file_list[idx]
        lb_file = self.lb_file_list[idx]



        ev = np.load(ev_file)['ev']
        lb = np.load(lb_file)['lb']
        events = rfn.structured_to_unstructured(ev)[:, [1, 2, 0, 3]]  # (x, y, t, p)
        labels = rfn.structured_to_unstructured(lb)[:, [1, 2, 3, 4, 5]]  # (x,y,w,h,class_id)
        events = self.downsample_event_stream(events)


        xs, ys, ts, ps = self.event_formatting(events[:, 0], events[:, 1], events[:, 2], events[:, 3])
        inp_cnt = self.create_cnt_encoding(xs, ys, ts, ps)
        inp_voxel = self.create_voxel_encoding(xs, ys, ts, ps)


        inp_list = self.create_list_encoding(xs, ys, ts, ps)
        inp_pol_mask = self.create_polarity_mask(ps)

        # hot pixel filter
        hot_mask = torch.ones((2,self.resize[0],self.resize[1]),dtype=bool)
        max = torch.quantile(inp_cnt, q=0.998)
        hot_mask[inp_cnt>max]=0
        inp_cnt *= hot_mask
        inp_voxel *= (hot_mask[0,:,:]*hot_mask[1,:,:])

        labels = self.cropToFrame(labels)
        labels = self.filter_boxes(labels, 60, 20)  # filter small boxes

        # # (x,y,w,h, class)->(x1,y1,x2,y2, class)
        labels[:, 2] += labels[:, 0]
        labels[:, 3] += labels[:, 1] # [x1, y1, x2, y2, class]

        # labels:(1280,720)->labels:(512,512)
        labels[:, 0] *= (self.resize[1]/self.width)
        labels[:, 1] *= (self.resize[0]/self.height)
        labels[:, 2] *= (self.resize[1]/self.width)
        labels[:, 3] *= (self.resize[0]/self.height)

        # input['ev'] = torch.stack([ts, xs, ys, ps]).transpose(1,0) # tensor(N,4) ts, xs, ys, ps
        input['lb'] = torch.tensor(labels,dtype=torch.float32)
        input['inp_cnt'] = inp_cnt  # tensor (2,h,w)
        input['inp_voxel'] = inp_voxel # tensor(5,360,640)
        input['inp_list'] = inp_list.transpose(1,0)
        input['inp_pol_mask'] = inp_pol_mask.transpose(1,0)
        input['hot_mask'] = hot_mask
        input['ev_file_name']=ev_file
        input['lb_file_name'] = lb_file
        # input['file_name'] = self.ev_file_list[idx]

        return input

    def __len__(self):
        return len(self.ev_file_list)

    @staticmethod
    def custom_collate(batch):
        """
        Collects the different event representations and stores them together in a dictionary.
        """

        batch_dict = {}
        for key in batch[0].keys():
            batch_dict[key] = []
        for entry in batch:
            for key in entry.keys():
                batch_dict[key].append(entry[key])
        batch_dict['inp_cnt']= torch.stack(batch_dict['inp_cnt'])
        batch_dict['inp_voxel'] = torch.stack(batch_dict['inp_voxel'])
        batch_dict['inp_list'] = torch.nn.utils.rnn.pad_sequence(batch_dict['inp_list'], batch_first=True)
        batch_dict['inp_pol_mask'] =torch.nn.utils.rnn.pad_sequence(batch_dict['inp_pol_mask'], batch_first=True)
        batch_dict['hot_mask'] = torch.stack(batch_dict['hot_mask'])

        return batch_dict


    def cropToFrame(self, np_bbox):
        """Checks if bounding boxes are inside frame. If not crop to border"""
        boxes = []
        for box in np_bbox:
            if box[2] > 1280:
                continue
            if box[0] < 0:
                box[2] += box[0]
                box[0] = 0
            if box[1] < 0:
                box[3] += box[1]
                box[1] = 0
            if box[0] + box[2] > self.width:
                box[2] = self.width - box[0]
            if box[1] + box[3] > self.height:
                box[3] = self.height - box[1]

            if box[2] > 0 and box[3] > 0 and box[0] < self.width and box[1] <= self.height:
                boxes.append(box)
        boxes = np.array(boxes).reshape(-1, 5)
        return boxes

    def filter_boxes(self, boxes, min_box_diag=60, min_box_side=20):
        """Filters boxes according to the paper rule.
        To note: the default represents our threshold when evaluating GEN4 resolution (1280x720)
        To note: we assume the initial time of the video is always 0
        :param boxes: (np.ndarray)
                     structured box array with fields ['t','x','y','w','h','class_id','track_id','class_confidence']
                     (example BBOX_DTYPE is provided in src/box_loading.py)
        Returns:
            boxes: filtered boxes
        """
        width = boxes[:, 2]
        height = boxes[:, 3]
        diag_square = width ** 2 + height ** 2
        mask = (diag_square >= min_box_diag ** 2) * (width >= min_box_side) * (height >= min_box_side)
        return boxes[mask]

    def downsample_event_stream(self, events):
        events[:, 0] = events[:, 0] / 1280 * self.resize[1]  # x
        events[:, 1] = events[:, 1] / 720 * self.resize[0]  # y

        if events[:,0].max()>=self.resize[1]:
            events[:, 0][np.where(events[:, 0]>=self.resize[1])]=self.resize[1]-1
        if events[:,1].max()>=self.resize[0]:
            events[:, 1][np.where(events[:, 1]>=self.resize[0])]=self.resize[0]-1

        return events





















