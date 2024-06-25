import os
import shutil

# 源文件夹和目标文件夹路径
source_base_path = '/home/wiser-renjie/remote_datasets/MOT17_Det_YOLO/datasets_separated'

# 比例设置
train_ratio = 0.5
val_ratio = 0.1
test_ratio = 0.4

seqs = ['MOT17-02-SDP',
        'MOT17-04-SDP',
        'MOT17-05-SDP',
        'MOT17-09-SDP',
        'MOT17-10-SDP',
        'MOT17-11-SDP',
        'MOT17-13-SDP']

# 遍历每个文件夹
for seq in seqs:
    folder_name = seq
    images_path = os.path.join(source_base_path, folder_name, 'images')
    labels_path = os.path.join(source_base_path, folder_name, 'labels')
    destination_base_path = os.path.join('/home/wiser-renjie/remote_datasets/MOT17_Det_YOLO/datasets_separated_splitted', folder_name)
    
    # 创建目标文件夹
    os.makedirs(os.path.join(destination_base_path, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(destination_base_path, 'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(destination_base_path, 'val/images'), exist_ok=True)
    os.makedirs(os.path.join(destination_base_path, 'val/labels'), exist_ok=True)
    os.makedirs(os.path.join(destination_base_path, 'test/images'), exist_ok=True)
    os.makedirs(os.path.join(destination_base_path, 'test/labels'), exist_ok=True)
    
    # 获取所有的图像和标签文件名
    image_files = sorted(os.listdir(images_path))
    label_files = sorted(os.listdir(labels_path))

    # 确保每个文件夹下的图像和标签文件数量一致
    assert len(image_files) == len(label_files), f"{folder_name} contains different number of images and labels."
    
    # 计算各数据集大小
    num_total = len(image_files)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)
    num_test = num_total - num_train - num_val
    
    # 分割数据集
    train_files = image_files[:num_train]
    val_files = image_files[num_train:num_train + num_val]
    test_files = image_files[num_train + num_val:]
    
    # 移动文件到相应目录并重命名
    def copy_files(files, subset):
        for file in files:
            base_name = os.path.splitext(file)[0]
            image_file = f"{base_name}.jpg"
            label_file = f"{base_name}.txt"
            new_image_name = f'{image_file}'
            new_label_name = f'{label_file}'
            shutil.copy2(os.path.join(images_path, image_file), os.path.join(destination_base_path, f'{subset}/images', new_image_name))
            shutil.copy2(os.path.join(labels_path, label_file), os.path.join(destination_base_path, f'{subset}/labels', new_label_name))
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

print("Dataset split and files moved successfully.")
