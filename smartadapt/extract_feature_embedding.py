import os
import os.path as osp
import pickle
from utils import LoadImages
from utils import extrack_embedding

def run(dataloader, result_root, seq, imgsz):
    len_all = len(dataloader)
    start_frame = int(len_all / 2)
    for i, (img_path, img_filename, img, img0) in enumerate(dataloader):
        print(f'Processing {seq}: {img_filename}')
        if i >= start_frame:
            break

        embedding = extrack_embedding(img, layer_idx=6)
                
        result_filename = osp.join(result_root, f'{img_filename}_{seq}_{imgsz}.pkl')
        
        with open(result_filename, 'wb') as file: 
            pickle.dump(embedding, file)
        

def main(data_root, results_root, seqs, imgsz_list):
    for seq in seqs:
        for imgsz in imgsz_list:
            dataloader = LoadImages(osp.join(data_root, seq, 'img1'), imgsz)
            
            run(dataloader, results_root, seq, imgsz)


if __name__ == '__main__':
    data_root = '/home/wiser-renjie/remote_datasets/MOT17'
    result_root = '/home/wiser-renjie/projects/yolov8/smartadapt/results'
    
    seqs = ['MOT17-02-SDP',
            'MOT17-04-SDP',
            'MOT17-05-SDP',
            'MOT17-09-SDP',
            'MOT17-10-SDP',
            'MOT17-11-SDP',
            'MOT17-13-SDP']

    imgsz_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
    
    main(data_root=data_root,
         results_root=result_root,
         seqs=seqs,
         imgsz_list=imgsz_list)
