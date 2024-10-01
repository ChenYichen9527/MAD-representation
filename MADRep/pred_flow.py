import os
import time
import torch
from configs.parse import loadYaml
from dataloader.dataset_EMGA import PropDataset
from flow_models.model import FireFlowNet, EVFlowNet
from utils.utils_visualization import Visualization
from utils.iwe import  compute_pol_iwe
import cv2
import numpy as np




if __name__ == "__main__":

    configs_path = r'configs/pre_flow_configs.yaml'
    config = loadYaml(configs_path)

    device = 'cuda:0'
    model = EVFlowNet(config["model_flow"].copy(), 5).to(device)
    model.load_state_dict(torch.load(config['model_path'], map_location=device))
    model.eval()


    dataset =PropDataset(config,mode='test')
    loader =  torch.utils.data.DataLoader(dataset, batch_size=1,collate_fn=dataset.custom_collate,sampler=None)

    vis = Visualization(config)

    for n,inputs in enumerate(loader):
        # forward pass
        inp_voxel = inputs["inp_voxel"].to(device)
        inp_cnt = inputs["inp_cnt"].to(device)
        t0 = time.time()
        x = model(inp_voxel,inp_cnt )
        t1=time.time()
        print('flow_time:',1000*(t1-t0))
        inp_list = inputs["inp_list"].to(device)
        inp_pol_mask0 = inputs["inp_pol_mask"][:, :, 0:1].to(device)
        inp_pol_mask1 = inputs["inp_pol_mask"][:, :, 1:2].to(device)
        iwe = compute_pol_iwe(
            x["flow"][0],
            inp_list,
           config['train']['resize'],
            inp_pol_mask0,
            inp_pol_mask1,
            flow_scaling=640,
            round_idx=True,
        )

        vis.update(inputs=inputs,flow=None,iwe=None,brightness=None)

        iwe = iwe.detach().cpu()
        iwe *= inputs['hot_mask']
        iwe_npy = iwe.numpy().transpose(0, 2, 3, 1).reshape((360, 640, 2))
        iwe_npy = Visualization.events_to_image(iwe_npy)
        img = iwe_npy[:, :, np.newaxis]
        img = np.repeat(img, 3, axis=2)
        if inputs['lb'][0].numel():
            labels = np.array(inputs['lb'][0])
            for label in labels:
                cv2.rectangle(img, (int(label[0]), int(label[1])), (int(label[2]), int(label[3])), color=(0, 255, 0))
        cv2.imshow('img', img)
        cv2.resizeWindow("img ", (960, 540))
        key = cv2.waitKey(0)
        if key == ord(' '):
            cv2.waitKey(10)












