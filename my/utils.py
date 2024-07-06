import os
import cv2
import time
import numpy as np
import os.path as osp
from kalman_filter import KalmanFilter

class STrack(object):
    def __init__(self, tlwh):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = KalmanFilter()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        
    def predict(self):
        mean_state = self.mean.copy()
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)
        
    def update(self, tlwh):
        """
        Update a matched track
        """
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(self._tlwh))

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def bbox(self):
        x1 = self.mean[0] + self.mean[4]
        y1 = self.mean[1] + self.mean[5]
        a = self.mean[2] + self.mean[6]
        h = self.mean[3] + self.mean[7]
        w = a * h
        return np.array([x1 - w/2, y1 - h/2, w, h], dtype=np.float32)
    
    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)
    return d

def tlwh2tlbr(x):
    x = np.asarray(x)
    y = np.empty_like(x)
    
    if y.size == 0:
        return y
    
    if y.shape[1] == 4:  # [x, y, w, h]
        y[..., 0] = x[..., 0]
        y[..., 1] = x[..., 1]
        y[..., 2] = x[..., 0] + x[..., 2]
        y[..., 3] = x[..., 1] + x[..., 3]
    elif y.shape[1] == 6: # [cls, x, y, w, h, conf]
        y[..., 0] = x[..., 0]
        y[..., 1] = x[..., 1]
        y[..., 2] = x[..., 2]
        y[..., 3] = x[..., 1] + x[..., 3]
        y[..., 4] = x[..., 2] + x[..., 4]
        y[..., 5] = x[..., 5]
    else:
        raise NotImplementedError("Input shape not supported")
    
    return y

def tlbr2tlwh(x):
    y = np.empty_like(x)
    
    if y.size == 0:
        return y
    
    if y.shape[1] == 4:  # [x1, y1, x2, y2]
        y[..., 0] = x[..., 0]
        y[..., 1] = x[..., 1]
        y[..., 2] = x[..., 2] - x[..., 0]
        y[..., 3] = x[..., 3] - x[..., 1]  # height
    elif y.shape[1] == 6:  # [cls, x1, y1, x2, y2, conf]
        y[..., 0] = x[..., 0]
        y[..., 1] = x[..., 1]
        y[..., 2] = x[..., 2]
        y[..., 3] = x[..., 3] - x[..., 1]  # width
        y[..., 4] = x[..., 4] - x[..., 2]  # height
        y[..., 5] = x[..., 5]
    else:
        raise NotImplementedError("Input shape not supported")
    
    return y

def tlwh2xywhn(x, H, W):  
    y = np.empty_like(x)
    
    if y.size == 0:
        return y
    
    if y.shape[1] == 4:  # [x1, y1, w, h]
        y[..., 0] = np.clip((x[..., 0] + x[..., 2] / 2) / W, 0, 1)
        y[..., 1] = np.clip((x[..., 1] + x[..., 3] / 2) / H, 0, 1)
        y[..., 2] = np.clip(x[..., 2] / W, 0, 1)
        y[..., 3] = np.clip(x[..., 3] / H, 0, 1)
    elif y.shape[1] == 6:  # [cls, x1, y1, w, h, conf]
        y[..., 0] = x[..., 0]
        y[..., 1] = np.clip((x[..., 1] + x[..., 3] / 2) / W, 0, 1)
        y[..., 2] = np.clip((x[..., 2] + x[..., 4] / 2) / H, 0, 1)
        y[..., 3] = np.clip(x[..., 3] / W, 0, 1)
        y[..., 4] = np.clip(x[..., 4] / H, 0, 1)
        y[..., 5] = x[..., 5]
    return y


def compute_union(bboxes, img_size):      # img_size = (H, W)
    if not bboxes:
        return None
    
    bboxes = np.array(bboxes)
    x1 = np.min(bboxes[..., 0])
    y1 = np.min(bboxes[..., 1])
    x2 = np.max(bboxes[..., 2])
    y2 = np.max(bboxes[..., 3])
    
    H, W = img_size
    # Clip the union bbox to be within the image size
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W, x2)
    y2 = min(H, y2)
    
    return [x1, y1, x2, y2]


# def compute_union(bboxes1, bboxes2, img_size):      # img_size = (H, W)

    
#     return [x1, y1, x2, y2]

def bbox_to_blocks(union_region, block_size=128):
    x1, y1, x2, y2 = union_region

    # Calculate block indices
    start_block_x = x1 // block_size
    start_block_y = y1 // block_size
    end_block_x = (x2 + block_size - 1) // block_size  # ceil division
    end_block_y = (y2 + block_size - 1) // block_size  # ceil division

    block_region = (block_size * start_block_x, block_size * start_block_y, block_size * end_block_x, block_size * end_block_y)
    
    return block_region


def detection_rate_adjuster(cluster_num):
    if cluster_num > 4 :
        detection_rate = 8
    elif cluster_num > 3:
        detection_rate = 9
    elif cluster_num > 2:
        detection_rate = 10
    else:
        detection_rate = 11
    return detection_rate

def merge_bboxes(bboxes1, bboxes2):
    bboxes1 = np.array(bboxes1)
    bboxes2 = np.array(bboxes2)
    
    if bboxes1.size == 0:
        return bboxes2
    if bboxes2.size == 0:
        return bboxes1
    return np.vstack((bboxes1, bboxes2))

def revert_bboxes(detected_bboxes, packed_rectangles):
    reverted_bboxes = []
    for detected_bbox in detected_bboxes:
        cls, dx1, dy1, dx2, dy2, conf = detected_bbox
        for rect in packed_rectangles:
            px1, py1, px2, py2 = rect['x'], rect['y'], rect['x'] + rect['width'], rect['y'] + rect['height']
            if px1 <= dx1 < px2 and py1 <= dy1 < py2:
                ox1, oy1, ox2, oy2 = rect['original_bbox']
                offset_x = ox1 - px1
                offset_y = oy1 - py1
                reverted_bboxes.append([cls, dx1 + offset_x, dy1 + offset_y, dx2 + offset_x, dy2 + offset_y, conf])
                break
    return np.array(reverted_bboxes)

def check_boundary(bboxes, hard_regions):
    bboxes = np.array(bboxes)
    hard_regions = np.array(hard_regions)

    mask = np.ones(len(bboxes), dtype=bool)

    bx1, by1, bx2, by2 = bboxes[..., 1], bboxes[..., 2], bboxes[..., 3], bboxes[..., 4]
    rx1, ry1, rx2, ry2 = hard_regions[..., 0], hard_regions[..., 1], hard_regions[..., 2], hard_regions[..., 3]
    
    for i in range(len(hard_regions)):
        in_region = (bx1 >= rx1[i]) & (by1 >= ry1[i]) & (bx2 <= rx2[i]) & (by2 <= ry2[i])
        mask &= ~in_region

    return bboxes[mask]

def plot_cluster(img, hard_blocks):
    overlay = img.copy()
    alpha = 0.3
    
    for x1, y1, x2, y2 in hard_blocks:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
    
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    return img

def plot_grid(img, block_size=128):
    h, w, _ = img.shape
    
    for y in range(0, h, block_size):
        cv2.line(img, (0, y), (w, y), (255, 0, 0), 1)
    for x in range(0, w, block_size):
        cv2.line(img, (x, 0), (x, h), (255, 0, 0), 1)
    
    return img

def plot_bbox(img, bboxes, color=(0, 255, 0), thickness=2):
    bboxes = np.asarray(bboxes, dtype=np.int32)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

def plot_bbox_with_labels(img, bboxes, labels, thresh=0.5, color=(0, 255, 0), thickness=2):
    bboxes = np.asarray(bboxes, dtype=np.int32)
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox       
        label_color = (0, 255, 255) if label < thresh else color
        cv2.rectangle(img, (x1, y1), (x2, y2), label_color, thickness)
        
        label = f'{int(label*100)}%'
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, thickness=2)
    return img

def scale_bbox(bboxes, Hc, Wc, Ht, Wt):             # Hc: current heright, Ht: target height
    bboxes = np.asarray(bboxes, dtype=np.float32)
    if bboxes.size != 0:
        scale_x = Wt / Wc
        scale_y = Ht / Hc
        
        scales = np.array([scale_x, scale_y, scale_x, scale_y])
        
        if bboxes.shape[1] == 4:
            bboxes = bboxes * scales
        elif bboxes.shape[1] == 6:
            bboxes[:, 1:5] = bboxes[:, 1:5] * scales

    return bboxes

def get_iou(bbox1, bbox2):
    bbox1 = np.asarray(bbox1)
    bbox2 = np.asarray(bbox2)
    assert bbox1.shape == bbox2.shape
    
    if bbox1.shape[0] == 6:
        bbox1 = bbox1[1:5]
        bbox2 = bbox2[1:5]
    
    ax1, ay1, ax2, ay2 = bbox1
    bx1, by1, bx2, by2 = bbox2
    assert ax1 < ax2, (bbox1, bbox2)
    assert ay1 < ay2, (bbox1, bbox2)
    assert bx1 < bx2, (bbox1, bbox2)
    assert by1 < by2, (bbox1, bbox2)

    # determine the coordinates of the intersection rectangle
    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (ax2 - ax1) * (ay2 - ay1)
    bb2_area = (bx2 - bx1) * (by2 - by1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def find_FPs(bboxes, gts, thresh=0.5):
    FPs = []
    for bbox in bboxes:
        best_iou = 0
        for gt in gts:
            iou = get_iou(bbox, gt)
            if iou > best_iou:
                best_iou = iou
        if best_iou < thresh:
            FPs.append(bbox)
    return FPs

def get_best_iou(bboxes, gts):
    ious = []
    for bbox in bboxes:
        best_iou = 0
        for gt in gts:
            iou = get_iou(bbox, gt)
            if iou > best_iou:
                best_iou = iou
        ious.append(best_iou)
    return ious

def run_detector(img, model, H, W, queue):
    start = time.time()
    results = model.predict(img, save=True, imgsz=(H, W), classes=[0], conf=0.3)
    bboxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
    confs = results[0].boxes.conf.cpu().numpy().astype(np.float32)
    clses = results[0].boxes.cls.cpu().numpy().astype(np.int32)
    out = np.hstack((clses[:, None], bboxes, confs[:, None]))
    end = time.time()
    detection_time = (end - start)*1000
    queue.put((out, detection_time))
    
def filter_bbox(bboxes, conf_thresh=0.5):
    bboxes = np.asarray(bboxes)
    if bboxes.size == 0:
        return bboxes
    filtered_bboxes = bboxes[bboxes[:, 4] > conf_thresh]
    return filtered_bboxes

def compute_center_distance(bboxes1, bboxes2):
    bboxes1 = np.asarray(bboxes1)
    bboxes2 = np.asarray(bboxes2)
    centers1 = (bboxes1[..., [0, 1]] + bboxes1[..., [2, 3]]) / 2
    centers2 = (bboxes2[..., [0, 1]] + bboxes2[..., [2, 3]]) / 2
    distances = np.sqrt(((centers1[:, np.newaxis, :] - centers2[np.newaxis, :, :]) ** 2).sum(axis=2))
    return distances

def handle_boundary_conflicts(prev_hard_bboxes, curr_hard_bboxes, dist_thresh=20, iou_thresh=0.8, type='dist'):
    prev_hard_bboxes = np.asarray(prev_hard_bboxes)
    curr_hard_bboxes = np.asarray(curr_hard_bboxes)
    
    valid_bboxes = []

    if type in ['both', 'dist']:
        distances = compute_center_distance(prev_hard_bboxes, curr_hard_bboxes)

    if type in ['both', 'iou']:
        ious = np.array([[get_iou(prev_bbox, curr_bbox) for curr_bbox in curr_hard_bboxes] for prev_bbox in prev_hard_bboxes])

    for i, prev_bbox in enumerate(prev_hard_bboxes):
        for j, curr_bbox in enumerate(curr_hard_bboxes):
            if type == 'dist' and distances[i, j] <= dist_thresh:
                valid_bboxes.append(curr_bbox)
            elif type == 'iou' and ious[i, j] >= iou_thresh:
                valid_bboxes.append(curr_bbox)
            elif type == 'both' and distances[i, j] <= dist_thresh and ious[i, j] >= iou_thresh:
                valid_bboxes.append(curr_bbox)
    
    return np.array(valid_bboxes)

def is_in_hard_block(bbox, hard_blocks):
    x1, y1, x2, y2 = bbox[1:5]
    for block in hard_blocks:
        bx1, by1, bx2, by2 = block
        if x1 >= bx1 and y1 >= by1 and x2 <= bx2 and y2 <= by2:
            return True
    return False

def clip_bbox(bboxes, H, W):
    bboxes = np.asarray(bboxes)
    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, W)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, H)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, W)
    bboxes[:, 4] = np.clip(bboxes[:, 4], 0, H)
    return bboxes

# def handle_boundary_conflicts(bboxes, hard_blocks, H, W):
#     bboxes = np.asarray(bboxes)
#     hard_blocks = np.asarray(hard_blocks)
    
#     # Clip the bounding boxes to the image boundaries
#     bboxes[:, 1] = np.clip(bboxes[:, 1], 0, W)
#     bboxes[:, 2] = np.clip(bboxes[:, 2], 0, H)
#     bboxes[:, 3] = np.clip(bboxes[:, 3], 0, W)
#     bboxes[:, 4] = np.clip(bboxes[:, 4], 0, H)

#     valid_bboxes = np.array([bbox for bbox in bboxes if is_in_hard_block(bbox, hard_blocks)])
    
#     return valid_bboxes

def find_bbox_in_hard_region(bboxes, hard_blocks):
    return np.array([bbox for bbox in bboxes if is_in_hard_block(bbox, hard_blocks)])
    