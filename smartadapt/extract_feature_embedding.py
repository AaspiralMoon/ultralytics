import os
import os.path as osp
import pickle
from utils import LoadImages
from utils import extract_embedding


def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)
        
        
def run(dataloader, output_root, seq, imgsz):
    for i, (img_path, img_filename, img, img0) in enumerate(dataloader):
        print(f'Processing {seq}: {img_filename}')

        embedding = extract_embedding(img, layer_idx=18)
        
        result_filename = osp.join(output_root, f'{seq}_{img_filename}_yolov8n_{imgsz}.pkl')
        
        with open(result_filename, 'wb') as file: 
            pickle.dump(embedding, file)
        

def main(data_root, output_root, seqs, imgsz_list):
    for seq in seqs:
        for imgsz in imgsz_list:
            dataloader = LoadImages(osp.join(data_root, seq, 'img1'), imgsz)
            
            run(dataloader, output_root, seq, imgsz)


if __name__ == '__main__':
    # MOT17 test
    seqs = ['MOT17-01-SDP',
            'MOT17-03-SDP',
            'MOT17-06-SDP',
            'MOT17-07-SDP',
            'MOT17-08-SDP',
            'MOT17-12-SDP',
            'MOT17-14-SDP']
    
    data_root = '/home/wiser-renjie/remote_datasets/MOT17/images/test'
    output_root = '/home/wiser-renjie/projects/yolov8/smartadapt/features'
    mkdir_if_missing(output_root)

    # imgsz_list = [(1088, 608), (864, 480), (704, 384), (640, 352), (576, 320)]
    imgsz_list = [(1088, 608)]
    
    main(data_root=data_root,
         output_root=output_root,
         seqs=seqs,
         imgsz_list=imgsz_list)
