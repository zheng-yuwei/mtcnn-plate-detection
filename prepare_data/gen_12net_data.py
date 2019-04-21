# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import numpy.random as npr

from prepare_data.utils import square_IoU
from prepare_data.path_configure import PathConfiguration

path_config = PathConfiguration().config
point2_data_path = path_config.point2_train_txt_path
train_image_dir = path_config.images_dir
pos_save_dir = path_config.pnet_pos_dir
part_save_dir = path_config.pnet_part_dir
neg_save_dir = path_config.pnet_neg_dir
save_dir = path_config.pnet_dir
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

pos_file = open(path_config.pnet_pos_txt_path, 'w')
neg_file = open(path_config.pnet_neg_txt_path, 'w')
part_file = open(path_config.pnet_part_txt_path, 'w')
with open(point2_data_path, 'r') as point2_file:
    annotations = point2_file.readlines()
num = len(annotations)
print("%d pics in total" % num)
pos_idx = 0  # 已生成的正样本图片数量
old_pos_idx = -1
no_pos_pic = 0  # 统计没有截取到正样本的图片
no_pos_file_names = list()
im_path = None
neg_idx = 0  # 已生成的负样本图片数量
part_idx = 0  # 已生成的部分样本图片数量
image_idx = 0  # 已处理图片的数量（处理指，利用该图片生成样本）

""" 造训练数据，对每一张训练图片：
负样本（iou < 0.3，标签0）：50张，label=0
每个bbox附近的负样本（iou < 0.3，标签0）：随机5张，label=0
part样本（0.4 <= iou < 0.65，标签-1）或正样本（iou >= 0.65，标签1）：本来是最多随机20张，我加强了产生的概率
部分样本label=-1, 正样本label=1
landmark标签为-2，在gen_landmark.py中
"""
for annotation in annotations:
    # 统计没有截取到正样本的图片
    if old_pos_idx == pos_idx:
        no_pos_pic += 1
        print(im_path)
    else:
        old_pos_idx = pos_idx
    annotation = annotation.strip().split(' ')
    im_path = annotation[0]
    boxes = np.array(list(map(float, annotation[1:])), dtype=np.float32).reshape(-1, 4)
    if (image_idx + 1) % 100 == 0:
        print("done: %s, pos: %s, part: %s, neg: %s, no positive pic: %s" %
              (image_idx + 1, pos_idx, part_idx, neg_idx, no_pos_pic))
    image_idx += 1

    img = cv2.imread(os.path.join(train_image_dir, im_path))
    image_height, image_width, _ = img.shape
    
    # 随机截取图片，每张图片生成负样本（iou < 0.3，标签0）：50张
    neg_num = 0
    while neg_num < 50:
        # 截取的负样本区域大小
        size_neg = npr.randint(12, min(image_width, image_height) / 2)
        # 负样本的左上角坐标
        x_left_neg = npr.randint(0, image_width - size_neg)
        y_top_neg = npr.randint(0, image_height - size_neg)
        # 截取区域
        crop_box = np.array([x_left_neg, y_top_neg, x_left_neg + size_neg, y_top_neg + size_neg])
        # 计算对ground truth的bounding boxes进行方形校正后的IoU
        iou = square_IoU(crop_box, boxes)
        
        # 截取区域的图片数据
        cropped_im = img[y_top_neg: y_top_neg + size_neg, x_left_neg: x_left_neg + size_neg, :]
        # 截取图片resize为 12 * 12
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
        
        if np.max(iou) < 0.3:
            save_file = os.path.join(neg_save_dir, "%s.jpg" % neg_idx)
            neg_file.write(save_file + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            neg_idx += 1
            neg_num += 1
    
    # 遍历ground truth的bounding boxes，产生对应的negative、positive和part样本
    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x_left_truth, y_top_truth, x_right_truth, y_bottom_truth = box
        # ground truth的宽和高
        width_truth = x_right_truth - x_left_truth + 1
        height_truth = y_bottom_truth - y_top_truth + 1
        
        # 忽略小bounding boxes
        if max(width_truth, height_truth) < 20 or x_left_truth < 0 or y_top_truth < 0:
            continue
        
        # 在ground truth周围，截取5个IoU小于0.3的图片作为negative样本
        neg_num = 0
        try_num = 0
        while neg_num < 5 and try_num < 50:
            size_neg = npr.randint(12, min(image_width, image_height) / 2)
            # 随机生成负样本左上角坐标相对ground truth的偏移量
            delta_x_neg = npr.randint(max(-size_neg, -x_left_truth), width_truth)
            delta_y_neg = npr.randint(max(-size_neg, -y_top_truth), height_truth)
            # 计算负样本左上角坐标
            x_left_neg = int(x_left_truth + delta_x_neg)
            y_top_neg = int(y_top_truth + delta_y_neg)
            # 若截取区域越界
            if x_left_neg + size_neg > image_width or y_top_neg + size_neg > image_height:
                try_num += 1
                continue
            crop_box = np.array([x_left_neg, y_top_neg, x_left_neg + size_neg, y_top_neg + size_neg])
            iou = square_IoU(crop_box, boxes)
            
            if np.max(iou) < 0.3:
                # 生成负样本图片
                cropped_im = img[y_top_neg: y_top_neg + size_neg, x_left_neg: x_left_neg + size_neg, :]
                resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
                save_file = os.path.join(neg_save_dir, "%s.jpg" % neg_idx)
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                neg_idx += 1
                neg_num += 1
            else:
                try_num += 1
        
        # 在ground truth周围生成对应的positive和part样本
        pos_part_num = 0
        try_num = 0
        while pos_part_num < 20 and try_num < 200:
            # 随机产生样本尺寸、左上角坐标偏移量
            size_pos_part = npr.randint(int(min(width_truth, height_truth) * 0.8),
                                        np.ceil(1.25 * max(width_truth, height_truth)))
            delta_x_pos_part = npr.randint(-int(width_truth * 0.2), int(width_truth * 0.2))
            delta_y_pos_part = npr.randint(-int(height_truth * 0.2), int(height_truth * 0.2))
            # 计算样本的左上角、右下角坐标
            x_left_pos_part = int(max(x_left_truth + width_truth / 2 + delta_x_pos_part - size_pos_part / 2, 0))
            y_top_pos_part = int(max(y_top_truth + height_truth / 2 + delta_y_pos_part - size_pos_part / 2, 0))
            x_right_pos_part = x_left_pos_part + size_pos_part
            y_bottom_pos_part = y_top_pos_part + size_pos_part
            # 越界
            if x_right_pos_part > image_width or y_bottom_pos_part > image_height:
                try_num += 1
                continue
                
            crop_box = np.array([x_left_pos_part, y_top_pos_part, x_right_pos_part, y_bottom_pos_part])
            iou = square_IoU(crop_box, box.reshape(1, -1))
            if iou >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % pos_idx)
                write_file = pos_file
                pos_idx += 1
                label = 1
            elif iou >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg" % part_idx)
                write_file = part_file
                part_idx += 1
                label = -1
            else:
                try_num += 1
                continue
                
            pos_part_num += 1
            # 计算偏移比例（训练时周围bounding box的label
            offset_x1 = (x_left_truth - x_left_pos_part) / float(size_pos_part)
            offset_y1 = (y_top_truth - y_top_pos_part) / float(size_pos_part)
            offset_x2 = (x_right_truth - x_right_pos_part) / float(size_pos_part)
            offset_y2 = (y_bottom_truth - y_bottom_pos_part) / float(size_pos_part)
            write_file.write(save_file +
                             ' %d %.2f %.2f %.2f %.2f\n' % (label, offset_x1, offset_y1, offset_x2, offset_y2))
            # 保存截取样本图片
            cropped_im = img[y_top_pos_part: y_bottom_pos_part, x_left_pos_part: x_right_pos_part, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(save_file, resized_im)
    
pos_file.close()
neg_file.close()
part_file.close()
