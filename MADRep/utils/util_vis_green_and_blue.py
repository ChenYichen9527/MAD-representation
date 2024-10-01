import numpy as np
import torch

def render_img(x: np.ndarray, y: np.ndarray, pol: np.ndarray, t: np.ndarray, H: int, W: int) -> np.ndarray:
    x = x.squeeze()
    y = y.squeeze()
    pol = pol.squeeze()
    t = t.squeeze()
    x =x.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    t = t.cpu().detach().numpy()
    pol = pol.cpu().detach().numpy()


    assert x.size == y.size == pol.size
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

            index = W * ylim.astype('int') + \
                    xlim.astype('int')

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
    img_acc[idx_pos] = img_acc[idx_pos] / img_acc_max
    val_pos = img_acc[idx_pos]
    img_out[idx_pos] = np.stack((255 - val_pos * 255, 255 - val_pos * 255, np.ones_like(val_pos) * 255), axis=1)

    img_acc_min = img_acc.min()
    idx_neg = img_acc < 0
    img_acc[idx_neg] = img_acc[idx_neg] / img_acc_min
    val_neg = img_acc[idx_neg]
    img_out[idx_neg] = np.stack((np.ones_like(val_neg) * 255, 255 - val_neg * 255, 255 - val_neg * 255), axis=1)
    return img_out

def events_wrap(flow, event_list, res, flow_scaling=128, round_idx=True, polarity_mask=None):
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

    warped_events = event_list[:, :, 1:3] + (1 - event_list[:, :, 0:1]) * event_flow * flow_scaling  # (1,5000,2)

    return warped_events

