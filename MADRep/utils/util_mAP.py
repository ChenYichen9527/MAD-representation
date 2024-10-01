import os
import numpy as np
import json
import cv2 as cv

def xyhw2xyxy(xyhw):
    x1 = xyhw[0] - xyhw[2] / 2
    y1 = xyhw[1] - xyhw[3] / 2
    x2 = xyhw[0] + xyhw[2] / 2
    y2 = xyhw[1] + xyhw[3] / 2
    return [x1, y1, x2, y2]

def load_pt(json_file):
    # load pt from json files,
    # get N x 6, 0:4 bbox, 4:conf 5:class
    content = json.load(open(json_file, 'r', encoding="utf-8"))
    if content is None:
        return None
    targes = content['targets']
    tar_len = len(targes)
    targets_pt = []
    for i in range(tar_len):
        # print(targes[i])
        classid = []
        conf = []
        conf.append(targes[i]['conf'])
        classid.append(targes[i]['classid'])
        x1y1x2y2 = xyhw2xyxy(targes[i]['rect'])
        targets_pt.append(x1y1x2y2 + conf + classid)
    return np.array(targets_pt)


def load_gt(txt_file, H, W):
    # load gt from txt files, yolo format,
    # get N x 5, 0: label 1:5 bbox
    f = open(txt_file)
    lines = f.readlines()
    targets = []
    for line in lines:
        contents = line.strip(" ").strip("\n").split(" ")
        contents = [float(x) for x in contents]
        # print(contents)
        class_id = contents[0]
        cx = contents[1] * W
        cy = contents[2] * H
        pw = contents[3] * W
        ph = contents[4] * H
        x1y1x2y2 = xyhw2xyxy([cx, cy, pw, ph])
        targets.append([class_id, x1y1x2y2[0], x1y1x2y2[1], x1y1x2y2[2], x1y1x2y2[3]])
    return np.array(targets)


def calc_iou(bbox1, bbox2):
    if not isinstance(bbox1, np.ndarray):
        bbox1 = np.array(bbox1)
    if not isinstance(bbox2, np.ndarray):
        bbox2 = np.array(bbox2)
    xmin1, ymin1, xmax1, ymax1, = np.split(bbox1, 4, axis=-1)
    xmin2, ymin2, xmax2, ymax2, = np.split(bbox2, 4, axis=-1)
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    ymin = np.maximum(ymin1, np.squeeze(ymin2, axis=-1))
    xmin = np.maximum(xmin1, np.squeeze(xmin2, axis=-1))
    ymax = np.minimum(ymax1, np.squeeze(ymax2, axis=-1))
    xmax = np.minimum(xmax1, np.squeeze(xmax2, axis=-1))
    h = np.maximum(ymax - ymin, 0)
    w = np.maximum(xmax - xmin, 0)

    intersect = h * w
    union = area1 + np.squeeze(area2, axis=-1) - intersect
    return intersect / union


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections
    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        # print(" pred_cls = ", pred_cls)
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)
            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            # print(" recall = ", recall)
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases
            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5
    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))
    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
    return ap, mpre, mrec


## 评测一张图片
def value_one(pts, gts):
    iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.shape[0]
    correct = np.zeros((pts.shape[0], niou), dtype=np.bool_)
    nl = len(gts)
    tcls_tensor = gts[:, 0]
    tcls = gts[:, 0].tolist() if nl else []  # target class
    for cls in np.unique(tcls_tensor):
        ti = (cls == (gts[:, 0]).astype(np.uint8)).nonzero()[0]  # prediction indices
        pi = (cls == (pts[:, 5]).astype(np.uint8)).nonzero()[0]  # target indices
        if len(pi > 0):
            IOUS = calc_iou(pts[pi, :4], gts[ti, 1:5])
            ious = IOUS.max(1)
            iid = IOUS.argmax(1)
            detected = []
            detected_set = set()
            for j in ((ious > iouv[0]).nonzero()[0]):
                d = ti[iid[j]]
                if d.item() not in detected_set:
                    detected_set.add(d.item())
                    detected.append(d)
                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                    if len(detected) == nl:  # all targets already located in image
                        break
    return correct, pts[:, 4], pts[:, 5], tcls



