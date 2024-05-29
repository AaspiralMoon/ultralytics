import os
import numpy as np

def read_txt(file_path):
    """
    Read a .txt file and return the content as a list of lists.
    Each inner list represents a bounding box in the format [class, x_center, y_center, w, h].
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    bboxes = []
    for line in lines:
        bbox = list(map(float, line.strip().split()))
        bboxes.append(bbox)
    return bboxes

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in the format [class, x_center, y_center, w, h] (normalized).
    """
    _, x1, y1, w1, h1 = box1
    _, x2, y2, w2, h2 = box2

    x1_min = x1 - w1 / 2
    y1_min = y1 - h1 / 2
    x1_max = x1 + w1 / 2
    y1_max = y1 + h1 / 2

    x2_min = x2 - w2 / 2
    y2_min = y2 - h2 / 2
    x2_max = x2 + w2 / 2
    y2_max = y2 + h2 / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = w1 * h1
    box2_area = w2 * h2

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def calculate_precision_recall(gt_boxes, pred_boxes, iou_threshold):
    tp = 0
    fp = 0
    fn = 0

    matched_gt = []
    for pred in pred_boxes:
        best_iou = 0
        best_gt = None
        for gt in gt_boxes:
            if gt[0] == pred[0]:  # Check if classes match
                iou = compute_iou(gt, pred)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.append(best_gt)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn

def calculate_ap(gt_boxes, pred_boxes, iou_threshold):
    precisions = []
    recalls = []

    tp, fp, fn = calculate_precision_recall(gt_boxes, pred_boxes, iou_threshold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    precisions.append(precision)
    recalls.append(recall)

    # Compute AP as the area under the precision-recall curve
    precisions = [0] + precisions + [0]
    recalls = [0] + recalls + [1]
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    ap = 0
    for i in range(1, len(precisions)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]

    return ap

def calculate_map(gt_dir, pred_dir, iou_threshold=0.5):
    gt_files = os.listdir(gt_dir)
    pred_files = os.listdir(pred_dir)

    assert len(gt_files) == len(pred_files), "Number of ground truth files and prediction files must be the same."

    aps = []
    for gt_file, pred_file in zip(gt_files, pred_files):
        gt_boxes = read_txt(os.path.join(gt_dir, gt_file))
        pred_boxes = read_txt(os.path.join(pred_dir, pred_file))

        ap = calculate_ap(gt_boxes, pred_boxes, iou_threshold)
        aps.append(ap)

    mAP = np.mean(aps)
    return mAP

# Example usage
gt_dir = "/home/wiser-renjie/remote_datasets/wildtrack/datasets_separated/C1/test"
pred_dir = "/home/wiser-renjie/projects/yolov8/my/runs/my/C1"
mAP = calculate_map(gt_dir, pred_dir, iou_threshold=0.5)
print(f"mAP: {mAP:.4f}")
