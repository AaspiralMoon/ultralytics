import numpy as np
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

def tlwh2tlbr(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)
    if y.size != 0:
        y[..., 0] = x[..., 0]
        y[..., 1] = x[..., 1]
        y[..., 2] = x[..., 0] + x[..., 2]
        y[..., 3] = x[..., 1] + x[..., 3]
    return y

def tlbr2tlwh(x):
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)
    if y.size != 0:
        y[..., 0] = x[..., 0]
        y[..., 1] = x[..., 1]
        y[..., 2] = x[..., 2] - x[..., 0]  # width
        y[..., 3] = x[..., 3] - x[..., 1]  # height
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

def bbox_to_blocks(union_region, block_size):
    x1, y1, x2, y2 = union_region

    # Calculate block indices
    start_block_x = x1 // block_size
    start_block_y = y1 // block_size
    end_block_x = (x2 + block_size - 1) // block_size  # ceil division
    end_block_y = (y2 + block_size - 1) // block_size  # ceil division

    block_region = (block_size * start_block_x, block_size * start_block_y, block_size * end_block_x, block_size * end_block_y)
    
    return block_region

def tlwh2xywhn(x, H, W):  
    y = np.empty_like(x)
    if y.size != 0:
        y[..., 0] = (x[..., 0] + x[..., 2] / 2) / W
        y[..., 1] = (x[..., 1] + x[..., 3] / 2) / H
        y[..., 2] = x[..., 2] / W
        y[..., 3] = x[..., 3] / H
    return y

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
        dx1, dy1, dx2, dy2 = detected_bbox
        for rect in packed_rectangles:
            px1, py1, px2, py2 = rect['x'], rect['y'], rect['x'] + rect['width'], rect['y'] + rect['height']
            if px1 <= dx1 < px2 and py1 <= dy1 < py2:
                ox1, oy1, ox2, oy2 = rect['original_bbox']
                offset_x = ox1 - px1
                offset_y = oy1 - py1
                reverted_bboxes.append([dx1 + offset_x, dy1 + offset_y, dx2 + offset_x, dy2 + offset_y])
                break
    return np.array(reverted_bboxes)

def check_boundary(bboxes, hard_regions):
    
    bboxes = np.array(bboxes)
    hard_regions = np.array(hard_regions)

    mask = np.ones(len(bboxes), dtype=bool)

    bx1, by1, bx2, by2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
    rx1, ry1, rx2, ry2 = hard_regions[..., 0], hard_regions[..., 1], hard_regions[..., 2], hard_regions[..., 3]
    
    for i in range(len(hard_regions)):
        in_region = (bx1 >= rx1[i]) & (by1 >= ry1[i]) & (bx2 <= rx2[i]) & (by2 <= ry2[i])
        mask &= ~in_region

    return bboxes[mask]