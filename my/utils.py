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
    
def compute_union(bboxes, img_size):      # img_size = (H, W)
    if not bboxes:
        return None
    
    bboxes = np.array(bboxes)
    x1 = np.min(bboxes[:, 0])
    y1 = np.min(bboxes[:, 1])
    x2 = np.max(bboxes[:, 2])
    y2 = np.max(bboxes[:, 3])
    
    # Clip the union bbox to be within the image size
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_size[1], x2)
    y2 = min(img_size[0], y2)
    
    return (x1, y1, x2, y2)

def bbox_to_blocks(union_region, block_size):
    x1, y1, x2, y2 = union_region
    block_w, block_h = block_size

    # Calculate block indices
    start_block_x = x1 // block_w
    start_block_y = y1 // block_h
    end_block_x = (x2 + block_w - 1) // block_w  # ceil division
    end_block_y = (y2 + block_h - 1) // block_h  # ceil division

    # # Generate list of block coordinates
    # blocks = []
    # for bx in range(start_block_x, end_block_x):
    #     for by in range(start_block_y, end_block_y):
    #         blocks.append((bx, by))
    block_region = (block_size * start_block_x, block_size * start_block_y, block_size * end_block_x, block_size * end_block_y)
    
    return block_region

def merge_region(img, block_regions):
    if len(block_regions) == 1:
        return block_regions[0]
    