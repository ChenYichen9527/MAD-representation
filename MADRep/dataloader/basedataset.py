import torch
import numpy as np
from utils.utils_encoding import events_to_voxel

class BaseDataLoader(torch.utils.data.Dataset):
    """
    Base class for dataloader.
    """

    def __init__(self, configs):
        self.configs = configs
        self.root = configs['dataset']['dataset_path']

        self.height = configs['dataset']['height']
        self.width = configs['dataset']['width']
        self.resize = configs['train']['resize']
        self.batch_size = configs['train']['batch_size']
        self.downsample = configs['train']['downsample']

        self.num_bins = self.configs["dataset"]["num_bins"]



    @staticmethod
    def create_polarity_mask(ps):
        """
        Creates a two channel tensor that acts as a mask for the input event list.
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 2] event representation
        """
        inp_pol_mask = torch.stack([ps, ps])
        inp_pol_mask[0, :][inp_pol_mask[0, :] < 0] = 0
        inp_pol_mask[1, :][inp_pol_mask[1, :] > 0] = 0
        inp_pol_mask[1, :] *= -1
        return inp_pol_mask
    @staticmethod
    def create_list_encoding(xs, ys, ts, ps):
        """
        Creates a four channel tensor with all the events in the input partition.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 4] event representation
        """

        return torch.stack([ts, ys, xs, ps])
    @staticmethod
    def event_formatting(xs, ys, ts, ps):
        """
        Reset sequence-specific variables.
        :param xs: [N] numpy array with event x location
        :param ys: [N] numpy array with event y location
        :param ts: [N] numpy array with event timestamp
        :param ps: [N] numpy array with event polarity ([-1, 1])
        :return xs: [N] tensor with event x location
        :return ys: [N] tensor with event y location
        :return ts: [N] tensor with normalized event timestamp
        :return ps: [N] tensor with event polarity ([-1, 1])
        """

        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(ys.astype(np.float32))
        ts = torch.from_numpy(ts.astype(np.float32))
        ps = torch.from_numpy(ps.astype(np.float32)) * 2 - 1
        ts = (ts - ts[0]) / (ts[-1] - ts[0])
        return xs, ys, ts, ps



    def create_cnt_encoding(self, xs, ys, ts, ps):
        """
        Creates a per-pixel and per-polarity event count representation.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [2 x H x W] event representation
        """

        return self.events_to_channels(xs, ys, ps, self.resize)

    def create_voxel_encoding(self, xs, ys, ts, ps):
        """
        Creates a spatiotemporal voxel grid tensor representation with a certain number of bins,
        as described in Section 3.1 of the paper 'Unsupervised Event-based Learning of Optical Flow,
        Depth, and Egomotion', Zhu et al., CVPR'19..
        Events are distributed to the spatiotemporal closest bins through bilinear interpolation.
        Positive events are added as +1, while negative as -1.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [B x H x W] event representation
        """

        return events_to_voxel(
            xs,
            ys,
            ts,
            ps,
            self.num_bins,
            sensor_size=self.configs["train"]["resize"],
        )




    def events_to_channels(self,xs, ys, ps, sensor_size=(180, 240)):
        """
        Generate a two-channel event image containing event counters.
        """

        assert len(xs) == len(ys) and len(ys) == len(ps)

        mask_pos = ps.clone()
        mask_neg = ps.clone()
        mask_pos[ps < 0] = 0
        mask_neg[ps > 0] = 0

        pos_cnt = self.events_to_image(xs, ys, ps * mask_pos, sensor_size=sensor_size)
        neg_cnt = self.events_to_image(xs, ys, ps * mask_neg, sensor_size=sensor_size)

        return torch.stack([pos_cnt, neg_cnt])


    def events_to_image(self,xs, ys, ps, sensor_size=(180, 240)):
        """
        Accumulate events into an image.
        """

        device = xs.device
        img_size = list(sensor_size)
        img = torch.zeros(img_size).to(device)

        if xs.dtype is not torch.long:
            xs = xs.long().to(device)
        if ys.dtype is not torch.long:
            ys = ys.long().to(device)
        img.index_put_((ys, xs), ps, accumulate=True)

        return img