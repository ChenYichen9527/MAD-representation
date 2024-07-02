# preprocess 1 Mpx

import argparse
import numpy as np
import os
from glob import glob
from dataloader.prophesee.src.io.psee_loader import PSEELoader
from dataloader.pseelabelloader import PSEElabel
from tqdm import tqdm

def build(args):
    root = args.root
    save_root = args.save_root
    detT = args.detT
    skip = args.skip

    dat_list = os.listdir(root)
    dat_path_list = []
    for dat in dat_list:
        dat_path = os.path.join(root, dat)
        if 'test' in dat:
            dat_path = os.path.join(dat_path, 'test')
        if 'val' in dat:
            dat_path = os.path.join(dat_path, 'val')
        if 'train' in dat:
            dat_path = os.path.join(dat_path, 'train')
        dat_path_list.append(dat_path)

    for dat in dat_path_list:

        data_path = glob(os.path.join(dat, '*.dat'))
        label_path = glob(os.path.join(dat, '*.npy'))


        for i, data_ in enumerate(data_path):
            name = data_.split('/')[-1].split('_td.dat')[0]
            ev_name = name + '_td.dat'
            lb_name = name + '_bbox.npy'
            # print(name)
            ev_path = os.path.join(dat, ev_name)
            lb_path = os.path.join(dat, lb_name)
            event_videos = PSEELoader(ev_path)
            box_videos = PSEElabel(lb_path, FPS=1000000 / detT)  # 按照50ms加载，并且跳过前20帧

            time = event_videos.total_time()
            num_frame = int(time / detT) - skip

            sava_name = data_.split('gen4')[-1].split('moorea')[0]
            ev_save_path = save_root + sava_name
            ev_save_path = os.path.join(ev_save_path, name, 'events')
            lb_save_path = save_root + sava_name
            lb_save_path = os.path.join(lb_save_path, name, 'labels')
            os.makedirs(ev_save_path, exist_ok=True)
            os.makedirs(lb_save_path, exist_ok=True)
            pbar = tqdm(range(num_frame))

            for idx in pbar:
                pbar.set_description(f"num {i + 1}/{len(data_path)}")
                idx += skip
                ev_save_path_ = os.path.join(ev_save_path, name + '_ev' + str(idx) + '.npz')
                lb_save_path_ = os.path.join(lb_save_path, name + '_lb' + str(idx) + '.npz')
                if os.path.exists(ev_save_path_) and os.path.exists(lb_save_path_):
                    continue
                # ev_save_path = os.path.join(r'/media/yons/新加卷/datasets/AOBnetDataset/moorea_2019-02-19_003_td_1525500000_1585500000/events','moorea_2019-02-19_003_td_1525500000_1585500000'+'_ev'+str(idx)+'.npz')
                # lb_save_path = os.path.join(r'/media/yons/新加卷/datasets/AOBnetDataset/moorea_2019-02-19_003_td_1525500000_1585500000/labels','moorea_2019-02-19_003_td_1525500000_1585500000'+'_lb'+str(idx)+'.npz')
                # label_t = box_videos.labels_ts[idx]
                # label = box_videos.get_label_at_t(label_t)
                try:
                    label_t = box_videos.labels_ts[idx]
                    label = box_videos.get_label_at_t(label_t)
                except:
                    print("out of labels")
                    continue

                event_videos.seek_time(label_t)
                events = event_videos.load_delta_t(detT)

                # x,y,t,p = events['x'],events['y'],events['t'],events['p']
                # events = np.array([x,y,t,p]).transpose(1,0)
                # x,y,w,h,cls,t = label['x'],label['y'],label['w'],label['h'],label['class_id'],label['t']
                np.savez_compressed(lb_save_path_, lb=label)
                np.savez_compressed(ev_save_path_, ev=events)


if __name__ == "__main__":
    pars = argparse.ArgumentParser()
    pars.add_argument('-r','--root',type =str,  help='the path of orin 1 mpx dataset',required=False)
    pars.add_argument('-sr','--save_root',type =str,  help='the save path of new 1 mpx dataset', required=False)
    pars.add_argument('-t', '--detT',type =int, help='event window size (ms)', default= 50,)
    pars.add_argument('-s', '--skip', type=int, help='skip the frame', default=10, )
    args = pars.parse_args()
    build(args)
    print(args)