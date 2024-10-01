import torch
import numpy as np


def purge_unfeasible(x, res):
    """
    Purge unfeasible event locations by setting their interpolation trained_models to zero.
    :param x: location of motion compensated events
    :param res: resolution of the image space
    :return masked indices
    :return mask for interpolation trained_models
    输入已经补偿过的事件点的位置
    """
    # print('idx1',x.shape)
    # mask = torch.ones((x.shape[0], x.shape[1], 1)).to(x.device)
    mask = torch.ones((x.shape[0], x.shape[1], 1),device=x.device)
    mask_y = (x[:, :, 0:1] < 0) + (x[:, :, 0:1] >= res[0])  # 将y中小于0和大于分辨率的光流去除 (1,20000,1)


    # print('mask_y.shape',mask_y.shape)
    # print('x.shape',x.shape)
    mask_x = (x[:, :, 1:2] < 0) + (x[:, :, 1:2] >= res[1])  # 将x中小于0和大于分辨率的光流去除
    mask[mask_y + mask_x] = 0

    return x * mask, mask

def get_iwe(events, flow, tref, res, flow_scaling, round_idx=False):
    """
    foor DESC get iwe
    """
    import time
    events=events.cpu().numpy()
    flow=flow.cpu().detach().numpy()

    t0=time.time()
    # event propagation
    # events = events.squeeze()
    warped_events = events[:, :, 1:3] + (tref - events[:, :, 0:1]) * flow * flow_scaling #(1,5000,2)
    warped_events = warped_events.squeeze()
    events = events.squeeze()
    x = warped_events[:,1]
    y=warped_events[:,0]
    t=events[:,0]
    pol=events[:,3]
    # H=440
    # W=640
    H = 240
    W = 304

    x = x.squeeze()
    y = y.squeeze()
    pol = pol.squeeze()
    t = t.squeeze()
    pol = (pol + 1) / 2
    assert x.shape == y.shape == pol.shape
    assert H > 0
    assert W > 0
    img_acc = np.zeros((H, W), dtype='float32').ravel()

    pol = pol.astype('int')
    x0 = x.astype('int')
    y0 = y.astype('int')
    t = t.astype('float64')
    value = 2 * pol - 1

    t_norm = (t - t.min()) / (t.max() - t.min())
    t_norm = t_norm ** 2
    t_norm = t_norm.astype('float32')
    assert t_norm.min() >= 0
    assert t_norm.max() <= 1

    for xlim in [x0, x0 + 1]:
        for ylim in [y0, y0 + 1]:
            mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0)
            interp_weights = value * (1 - np.abs(xlim - x)) * (1 - np.abs(ylim - y)) * t_norm

            index = W * ylim.astype('int') + xlim.astype('int')

            np.add.at(img_acc, index[mask], interp_weights[mask])

    img_acc = np.reshape(img_acc, (H, W))

    img_out = np.full((H, W, 3), fill_value=255, dtype='uint8')

    # Simple thresholding
    # img_out[img_acc > 0] = [0,0,255]
    # img_out[img_acc < 0] = [255,0,0]

    # With weighting (more complicated alternative)
    clip_percentile = 80
    min_percentile = -np.percentile(np.abs(img_acc[img_acc < 0]), clip_percentile)
    max_percentile = np.percentile(np.abs(img_acc[img_acc > 0]), clip_percentile)
    img_acc = np.clip(img_acc, min_percentile, max_percentile)

    img_acc_max = img_acc.max()
    idx_pos = img_acc > 0
    cnt_pos = img_acc * idx_pos
    # img_acc[idx_pos] = img_acc[idx_pos]
    # val_pos = img_acc[idx_pos]
    # img_out[idx_pos] = np.stack((255 - val_pos * 255, 255 - val_pos * 255, np.ones_like(val_pos) * 255), axis=1)

    img_acc_min = img_acc.min()
    idx_neg = img_acc < 0
    cnt_neg = img_acc * idx_neg * (-1)
    # img_acc[idx_neg] = img_acc[idx_neg] / img_acc_min
    # val_neg = img_acc[idx_neg]
    # img_out[idx_neg] = np.stack((np.ones_like(val_neg) * 255, 255 - val_neg * 255, 255 - val_neg * 255), axis=1)
    img_out = torch.stack([torch.tensor(cnt_pos), torch.tensor(cnt_neg)], dim=0)
    return img_out



def get_interpolation(events, flow, tref, res, flow_scaling, round_idx=False):
    """
    Warp the input events according to the provided optical flow map and compute the bilinar interpolation
    (or rounding) trained_models to distribute the events to the closes (integer) locations in the image space.
    :param events: [batch_size x N x 4] input events (y, x, ts, p) (ts,y,x,p)
    :param flow: [batch_size x 2 x H x W] optical flow map
    :param tref: reference time toward which events are warped
    :param res: resolution of the image space
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = False)
    :return interpolated event indices
    :return interpolation trained_models
    """
    import time
    t0=time.time()
    # event propagation
    warped_events = events[:, :, 1:3] + (tref - events[:, :, 0:1]) * flow * flow_scaling #(1,5000,2)

    if round_idx:

        # no bilinear interpolation
        idx = torch.round(warped_events)
        weights = torch.ones(idx.shape,device=events.device)
        # weights = torch.ones(idx.shape).to(events.device)


    else:

        # get scattering indices
        top_y = torch.floor(warped_events[:, :, 0:1])
        bot_y = torch.floor(warped_events[:, :, 0:1] + 1)
        left_x = torch.floor(warped_events[:, :, 1:2])
        right_x = torch.floor(warped_events[:, :, 1:2] + 1)

        top_left = torch.cat([top_y, left_x], dim=2)
        top_right = torch.cat([top_y, right_x], dim=2)
        bottom_left = torch.cat([bot_y, left_x], dim=2)
        bottom_right = torch.cat([bot_y, right_x], dim=2)
        idx = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=1)

        # get scattering interpolation trained_models
        warped_events = torch.cat([warped_events for i in range(4)], dim=1)
        zeros = torch.zeros(warped_events.shape).to(events.device)
        weights = torch.max(zeros, 1 - torch.abs(warped_events - idx))
    t2=time.time()
    # purge unfeasible indices 清除不可行的索引
    idx, mask = purge_unfeasible(idx, res) #去除光流wrap之后不在分辨率范围内的点
    # idx[:,:,0] = torch.clamp(idx[:,:,0],max=719) # 强制索引范围
    # print('idx2', idx[:,:,0].max(),idx[:,:,1].max())
    # print('res',res)
    t3=time.time()

    # make unfeasible trained_models zero
    weights = torch.prod(weights, dim=-1, keepdim=True) * mask  # bilinear interpolation

    # prepare indices
    idx[:, :, 0] *= res[1]  # torch.view is row-major
    idx = torch.sum(idx, dim=2, keepdim=True)
    return idx, weights


def interpolate(idx, weights, res, polarity_mask=None):
    """
    Create an image-like representation of the warped events.
    :param idx: [batch_size x N x 1] warped event locations
    :param weights: [batch_size x N x 1] interpolation trained_models for the warped events
    :param res: resolution of the image space
    :param polarity_mask: [batch_size x N x 2] polarity mask for the warped events (default = None)
    :return image of warped events
    """
    # idx  = idx.to('cpu')
    # weights = weights.to('cpu')
    import time
    t0=time.time()
    device = 'cuda:0'
    if polarity_mask is not None:
        polarity_mask = polarity_mask.to(device)
        weights = weights * polarity_mask

    iwe = torch.zeros((idx.shape[0], res[0] * res[1], 1),device=device)
    idx = idx.long().clip(0, iwe.shape[1] - 1)

    iwe = iwe.scatter_add(1, idx, weights) #iwe(1,720*1280,1)
    iwe = iwe.to(device)
    iwe = iwe.view((idx.shape[0], 1, res[0], res[1]))


    return iwe

def interpolate_cuda(idx, weights, res, polarity_mask=None):
    """
    Create an image-like representation of the warped events.
    :param idx: [batch_size x N x 1] warped event locations
    :param weights: [batch_size x N x 1] interpolation trained_models for the warped events
    :param res: resolution of the image space
    :param polarity_mask: [batch_size x N x 2] polarity mask for the warped events (default = None)
    :return image of warped events
    """
    if polarity_mask is not None:
        # polarity_mask = polarity_mask.to('cpu')
        weights = weights * polarity_mask
    iwe = torch.zeros((idx.shape[0], res[0] * res[1], 1)).to('cuda:0')
    idx = idx.long().clip(0,iwe.shape[1]-1)
    # print('idx:',idx.shape,idx.max())
    # print('weights',weights.shape,weights.max())
    iwe = iwe.scatter_add(1, idx, weights) #iwe(1,720*1280,1)
    iwe = iwe.view((idx.shape[0], 1, res[0], res[1]))
    return iwe


def deblur_events(flow, event_list, res, flow_scaling=128, round_idx=True, polarity_mask=None):
    """
    Deblur the input events given an optical flow map.
    Event timestamp needs to be normalized between 0 and 1.
    :param flow: [batch_size x 2 x H x W] optical flow map
    :param events: [batch_size x N x 4] input events (y, x, ts, p)
    :param res: resolution of the image space
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = False)
    :param polarity_mask: [batch_size x N x 2] polarity mask for the warped events (default = None)
    :return iwe: [batch_size x 1 x H x W] image of warped events
    """
    import time

    t0=time.time()
    # flow vector per input event
    flow_idx = event_list[:, :, 1:3].clone()
    flow_idx[:, :, 0] *= res[1]  # torch.view is row-major
    flow_idx = torch.sum(flow_idx, dim=2)

    # get flow for every event in the list
    flow = flow.view(flow.shape[0], 2, -1)
    event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
    event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
    event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
    event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
    event_flow = torch.cat([event_flowy, event_flowx], dim=2)


    # interpolate forward
    iwe = get_iwe(event_list, event_flow, 1, res, flow_scaling, round_idx=round_idx)

    # fw_idx, fw_weights = get_interpolation(event_list, event_flow, 1, res, flow_scaling, round_idx=round_idx)
    # if not round_idx:
    #     polarity_mask = torch.cat([polarity_mask for i in range(4)], dim=1)
    #
    # # image of (forward) warped events
    # iwe = interpolate(fw_idx.long(), fw_weights, res, polarity_mask=polarity_mask)


    return iwe


def compute_pol_iwe(flow, event_list, res, pos_mask, neg_mask, flow_scaling=128, round_idx=True):
    """
    Create a per-polarity image of warped events given an optical flow map.
    :param flow: [batch_size x 2 x H x W] optical flow map
    :param event_list: [batch_size x N x 4] input events (y, x, ts, p)
    :param res: resolution of the image space
    :param pos_mask: [batch_size x N x 1] polarity mask for positive events
    :param neg_mask: [batch_size x N x 1] polarity mask for negative events
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = True)
    :return iwe: [batch_size x 2 x H x W] image of warped events
    """

    iwe_pos = deblur_events(
        flow, event_list, res, flow_scaling=flow_scaling, round_idx=round_idx, polarity_mask=pos_mask
    )
    # iwe_neg = deblur_events(
    #     flow, event_list, res, flow_scaling=flow_scaling, round_idx=round_idx, polarity_mask=neg_mask
    # )
    # iwe = torch.cat([iwe_pos, iwe_neg], dim=1)

    return iwe_pos
