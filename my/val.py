import os
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def load_annotations(annotations_path):
    coco = COCO(annotations_path)
    return coco

def load_detections(detections_path):
    with open(detections_path, 'r') as f:
        detections = json.load(f)
    return detections

def evaluate(coco, detections):
    coco_dt = coco.loadRes(detections)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    annotations_path = '/home/wiser-renjie/projects/yolov8/my/runs/my/jsons/gt_MOT17-04-SDP.json'  # Path to your ground truth annotations in COCO format
    detections_path = '/home/wiser-renjie/projects/yolov8/my/runs/my/jsons/MOT17-04-SDP_medianflow_yolox_1152_1920_0.3_144_240_i5.json'   # Path to your precomputed results in COCO format

    coco = load_annotations(annotations_path)
    detections = load_detections(detections_path)
    evaluate(coco, detections)