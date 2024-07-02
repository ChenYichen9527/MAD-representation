import cv2
import numpy as np
import h5py
import hdf5plugin
import os
from DSEC_labels import labels_11
import shutil
import argparse
from tqdm import tqdm

"将原始的DSEC文件build为单张的图片以及标注"
def build(conf):
    # settings
    time_window = conf.detT *1000  # 50ms
    skip = conf.skip  # skip 10 imgs at beging
    W,H = 680,480
    vis = conf.vis

    # path
    root = conf.root  #r'/media/yons/G/dataset/DSEC-seg/DSEC_Semantic'
    save_root = conf.save_root  #r'/media/yons/G/dataset/DSEC-seg/github_test'
    test_root = os.path.join(root, 'test')
    train_root = os.path.join(root, 'train')

    # color map
    color = labels_11
    map_dict = {}
    for i in color:
        map_dict[str(i.trainId)]=i.color

    # build test datasets
    test_list = os.listdir(test_root)
    for test_file in test_list:
        ev_file = os.path.join(test_root, test_file, 'events/left/events.h5')
        # rec_file = os.path.join(test_root, test_file, 'events/left/rectify_map.h5')
        # with h5py.File(rec_file,'r') as ev:
        #     rectify_map = ev['rectify_map'][()]


        label_file = os.path.join(root, 'test_semantic_segmentation', test_file, '11classes')
        label_list = os.listdir(label_file)

        sem_ts_file = os.path.join(root, 'test_semantic_segmentation', test_file, (test_file + r'_semantic_exposure_timestamps_left.txt'))
        sem_ts = np.loadtxt(sem_ts_file,delimiter=',')
        sem_ts = sem_ts[:,0]

        print('loading h5py')
        with h5py.File(ev_file, 'r') as ev:
            x = np.array(ev['events/x'])
            y = np.array(ev['events/y'])
            t = np.array(ev['events/t'])
            p = np.array(ev['events/p'])
            # ms2idx = np.array(ev['ms_to_idx'])
            t_offset = np.array(ev['t_offset'])

        #align timestample
        sem_ts = sem_ts - t_offset

        events = np.stack([x, y, t, p], axis=1)

        pbar = tqdm(range(len(sem_ts)))
        for i in pbar:
            i = i+skip
            pbar.set_description(f"num {i}/{len(pbar)}")

            sem_t = sem_ts[i]
            start_t = sem_t - time_window
            end_t = sem_t
            idx = np.where((start_t < events[:,2]) & (events[:,2] < end_t))
            wind_t = events[:,2][idx]
            wind_x = events[:,0][idx]
            wind_y = events[:,1][idx]
            wind_p = events[:,3][idx]

            # rectify events
            # xy_rect = rectify_map[wind_y, wind_x]
            # wind_x = xy_rect[:, 0]
            # wind_y = xy_rect[:, 1]

            wind_events = np.stack([wind_x, wind_y,wind_t, wind_p], axis=1)
            wind_events = wind_events[(wind_events[:, 0] >= 0) & (wind_events[:, 0] < W) & (wind_events[:, 1] >= 0) & (wind_events[:, 1] < H)]

            ev_save_path = os.path.join(save_root,'test',test_file,'events')
            os.makedirs(ev_save_path,exist_ok=True)
            label_save_path = os.path.join(save_root, 'test', test_file, 'labels')
            os.makedirs(label_save_path, exist_ok=True)

            save_file = os.path.join(ev_save_path,(str(i).zfill(6)+".npy"))
            np.save(save_file,wind_events)
            label_img_file = os.path.join(label_file, (str(i).zfill(6) + ".png"))
            copy_label_img_file = os.path.join(label_save_path,(str(i).zfill(6)+".png"))
            shutil.copy(label_img_file,copy_label_img_file)



            if vis:
                labe_img_size = (480, 640,3)
                rgb_img = np.ones(shape=labe_img_size, dtype=np.uint8)*255
                label_img_file = os.path.join(label_file, (str(i).zfill(6) + ".png"))
                label_img = cv2.imread(label_img_file)
                label_img = cv2.cvtColor(label_img,cv2.COLOR_RGB2GRAY)
                for trainId in range(11):
                    mask = np.where(label_img==trainId)
                    rgb_img[mask]=[map_dict[str(trainId)]]

                cv2.imshow('label_img', rgb_img)

                img_size = (480, 640)
                img = np.zeros(shape=img_size, dtype=np.uint8)
                for i in range(len(wind_events)):
                    img[int(wind_events[:, 1][i]), int(wind_events[:, 0][i])] = 1
                img = img[0:480,:]
                img = img.astype(np.uint8) * 255
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.imshow('img', img)


                add = cv2.addWeighted(img,0.5,rgb_img,0.5,0)
                cv2.imshow('add',add)

                key = cv2.waitKey(1)
                # if key == ord(' '):
                #     cv2.waitKey(10)


    # build train_set
    test_root = train_root
    test_list = os.listdir(test_root)
    for test_file in test_list:
        ev_file = os.path.join(test_root, test_file, 'events/left/events.h5')
        rec_file = os.path.join(test_root, test_file, 'events/left/rectify_map.h5')
        with h5py.File(rec_file,'r') as ev:
            rectify_map = ev['rectify_map'][()]


        label_file = os.path.join(root, 'train_semantic_segmentation', test_file, '11classes')
        label_list = os.listdir(label_file)

        sem_ts_file = os.path.join(root, 'train_semantic_segmentation', test_file, (test_file + r'_semantic_exposure_timestamps_left.txt'))
        sem_ts = np.loadtxt(sem_ts_file,delimiter=',')
        sem_ts = sem_ts[:,1]

        print('loading h5py')
        with h5py.File(ev_file, 'r') as ev:
            x = np.array(ev['events/x'])
            y = np.array(ev['events/y'])
            t = np.array(ev['events/t'])
            p = np.array(ev['events/p'])
            # ms2idx = np.array(ev['ms_to_idx'])
            t_offset = np.array(ev['t_offset'])
        print('loaded h5py')

        #align timestample
        sem_ts = sem_ts - t_offset

        events = np.stack([x, y, t, p], axis=1)
        # events = events[(events[:,0]>=0) & (events[:,0]<640) & (events[:,1]>=0) & (events[:,1]<440) ]
        print('rec finish')

        for i in range(skip, len(sem_ts)):
            print('{}/{}'.format(i,len(sem_ts)))
            sem_t = sem_ts[i]
            start_t = sem_t - time_window
            end_t = sem_t
            idx = np.where((start_t < events[:,2]) & (events[:,2] < end_t))
            wind_t = events[:,2][idx]
            wind_x = events[:,0][idx]
            wind_y = events[:,1][idx]
            wind_p = events[:,3][idx]

            # rectify events
            xy_rect = rectify_map[wind_y, wind_x]
            # wind_x = xy_rect[:, 0].astype(int)
            # wind_y = xy_rect[:, 1].astype(int)
            wind_x = xy_rect[:, 0]
            wind_y = xy_rect[:, 1]

            wind_events = np.stack([wind_x, wind_y,wind_t, wind_p], axis=1)
            wind_events = wind_events[(wind_events[:, 0] >= 0) & (wind_events[:, 0] < 640) & (wind_events[:, 1] >= 0) & (wind_events[:, 1] < 440)]

            ev_save_path = os.path.join(save_root,'train',test_file,'events')
            os.makedirs(ev_save_path,exist_ok=True)
            label_save_path = os.path.join(save_root, 'train', test_file, 'labels')
            os.makedirs(label_save_path, exist_ok=True)

            save_file = os.path.join(ev_save_path,(str(i).zfill(6)+".npy"))
            np.save(save_file,wind_events)
            label_img_file = os.path.join(label_file, (str(i).zfill(6) + ".png"))
            copy_label_img_file = os.path.join(label_save_path,(str(i).zfill(6)+".png"))
            shutil.copy(label_img_file,copy_label_img_file)



            if vis:
                labe_img_size = (440, 640,3)
                rgb_img = np.ones(shape=labe_img_size, dtype=np.uint8)*255
                label_img_file = os.path.join(label_file, (str(i).zfill(6) + ".png"))
                label_img = cv2.imread(label_img_file)
                label_img = cv2.cvtColor(label_img,cv2.COLOR_RGB2GRAY)
                for trainId in range(11):
                    mask = np.where(label_img==trainId)
                    rgb_img[mask]=[map_dict[str(trainId)]]

                cv2.imshow('label_img', rgb_img)

                img_size = (480, 640)
                img = np.zeros(shape=img_size, dtype=np.uint8)
                for i in range(len(wind_events)):
                    img[int(wind_events[:, 1][i]), int(wind_events[:, 0][i])] = 1
                img = img[0:440,:]
                img = img.astype(np.uint8) * 255
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.imshow('event_count_img', img)


                add = cv2.addWeighted(img,0.5,rgb_img,0.5,0)
                cv2.imshow('add',add)

                key = cv2.waitKey(00)
                # if key == ord(' '):
                #     cv2.waitKey(10)


if __name__ == "__main__":
    pars = argparse.ArgumentParser()
    pars.add_argument('-r', '--root', type=str, help='the path of orin 1 mpx dataset', required=True)
    pars.add_argument('-sr', '--save_root', type=str, help='the save path of new 1 mpx dataset', required=True)
    pars.add_argument( '--detT', type=int, help='event window size (ms)', default=50, )
    pars.add_argument( '--skip', type=int, help='skip the frame', default=10, )
    pars.add_argument('--vis', type=bool, help='visualized flog', default=False, )
    args = pars.parse_args()
    build(args)


