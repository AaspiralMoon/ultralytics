import os
import numpy as np

if __name__ == '__main__':
    label_root = '/home/wiser-renjie/datasets/test_full/val/labels'
    
    for filename in sorted(os.listdir(label_root)):    
        label_path = os.path.join(label_root, filename)
        
        print(f'Processing: {label_path}')
         
        label = np.loadtxt(label_path)
        label[:, 0] = 0
        
        np.savetxt(label_path, label)