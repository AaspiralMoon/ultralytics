import numpy as np
import cv2
import time
from ultralytics import YOLO
from cython_bbox import bbox_overlaps as bbox_ious

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )
    return ious


def grow_cluster(D, labels, P, NeighborPts, C, eps, MinPts, IoU):

    # Assign the cluster label to the seed point.
    labels[P] = C
    
    i = 0
    while i < len(NeighborPts):    
        
        # Get the next point from the queue.        
        Pn = NeighborPts[i]
       
        # mark as noisy node
        if labels[Pn] == -1:
           labels[Pn] = C
        
        # Otherwise, if Pn isn't already claimed, claim it as part of C.
        elif labels[Pn] == 0:
            # Add Pn to cluster C (Assign cluster label C).
            labels[Pn] = C
            
            # Find all the neighbors of Pn
            PnNeighborPts = region_query(D, Pn, eps, IoU)
            
            if len(PnNeighborPts) >= MinPts:
                NeighborPts = NeighborPts + PnNeighborPts         
        
        # move to the next point
        i += 1        


def region_query(D, P, eps, IoU):

    neighbors = []
    
    # For each objects
    for Pn in range(0, len(D)):
        iou = ious([D[P][1]],[D[Pn][1]])
        # If the distance is below the threshold, and IoU is greater than the threshold add it to the neighbors list.
        if np.linalg.norm(D[P][0] - D[Pn][0]) < eps and iou > IoU:
           neighbors.append(Pn)
            
    return neighbors


def modified_dbscan(D, eps, MinPts, IoU):
    
    # Initially all labels are 0.    
    labels = [0]*len(D)

    # C is the ID of the current cluster.    
    C = 0
    
    for P in range(0, len(D)):
    
        # Only points that have not already been claimed can be picked as new 
        # seed points.    
        # If the point's label is not 0, continue to the next point.
        if not (labels[P] == 0):
           continue
        
        # Find all of P's neighboring points.
        NeighborPts = region_query(D, P, eps, IoU)
        
        if len(NeighborPts) < MinPts:
            labels[P] = -1
        # Otherwise, if there are at least MinPts nearby, use this point as the 
        # seed for a new cluster.    
        else: 
           C += 1
           grow_cluster(D, labels, P, NeighborPts, C, eps, MinPts, IoU)
    
    # All data has been clustered!
    return labels


def dbscan_clustering(bboxes):
    
    centroid_list = []
    tid_list = []
    tid = 0
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        tid_list.append(tid)
        tid += 1
        centroid_list.append((np.array([(x1 + x2) / 2, (y1 + y2) / 2]),[x1, y1, x2, y2]))

    cluster_label = modified_dbscan(centroid_list, 100, 3, 0.2)
    
    cluster_dic = {}
    cluster_num = 0
    for idx, each in enumerate(cluster_label):
        if each != -1 and each not in cluster_dic.keys():
            cluster_dic[each]=[tid_list[idx]]
            cluster_num += 1
        else:
            if each != -1:
                cluster_dic[each].append(tid_list[idx])
            else:
                cluster_num += 1
    return cluster_dic, cluster_num

# Load a pretrained YOLOv8n model
model = YOLO('yolov8x.pt')

img = cv2.imread('/home/wiser-renjie/remote_datasets/wildtrack/datasets_combined/train/images/C7_00001360.png')

results = model.predict(img, save_txt=False, save=False, classes=[0], imgsz=640, conf=0.5)

bboxes = results[0].boxes.xyxy.cpu().numpy()
scores = results[0].boxes.conf.cpu().numpy()

t1 = time.time()
cluster_dic, cluster_num = dbscan_clustering(bboxes)
t2 = time.time()
print(f'cluster time: {(t2-t1)*1000} ms')

print(cluster_dic, cluster_num)

