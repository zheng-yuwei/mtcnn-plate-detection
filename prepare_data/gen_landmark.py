# -*- coding: utf-8 -*-
import os
import random
import sys
import cv2
import numpy as np
import numpy.random as npr
import argparse

from prepare_data.BBox_utils import get_landmark_data, BBox
from prepare_data.Landmark_utils import rotate, flip
from prepare_data.utils import square_IoU, convert_to_square
from prepare_data.path_configure import PathConfiguration

path_config = PathConfiguration().config


def generate_landmark_data(landmark_truth_txt_path, images_dir, net, argument=False):
    """ 为特定网络类型生成关键点训练样本，label=-2
    :param landmark_truth_txt_path: 包含image path, bounding box, and landmarks的txt路径
    :param images_dir: 图片文件夹路径
    :param net: 网络类型,('PNet', 'RNet', 'ONet')
    :param argument: 是否进行数据增强
    :return:  images and related landmarks
    """
    if net == "PNet":
        size = 12
        landmark_dir = path_config.pnet_landmark_dir
        net_data_root_dir = path_config.pnet_dir
        landmark_file = open(path_config.pnet_landmark_txt_path, 'w')
    elif net == "RNet":
        size = 24
        landmark_dir = path_config.rnet_landmark_dir
        net_data_root_dir = path_config.rnet_dir
        landmark_file = open(path_config.rnet_landmark_txt_path, 'w')
    elif net == "ONet":
        size = 48
        landmark_dir = path_config.onet_landmark_dir
        net_data_root_dir = path_config.onet_dir
        landmark_file = open(path_config.onet_landmark_txt_path, 'w')
    else:
        raise ValueError('网络类型(--net)错误！')
    
    if not os.path.exists(net_data_root_dir):
        os.mkdir(net_data_root_dir)
    if not os.path.exists(landmark_dir):
        os.mkdir(landmark_dir)
    
    # 读取关键点信息文件：image path , bounding box, and landmarks
    data = get_landmark_data(landmark_truth_txt_path, images_dir)
    # 针对每张图片，生成关键点训练数据
    landmark_idx = 0
    image_id = 0
    for (imgPath, bbox, landmarkGt) in data:
        # 截取的图片数据和图片中关键点位置数据
        cropped_images = []
        cropped_landmarks = []

        img = cv2.imread(imgPath)
        assert (img is not None)
        image_height, image_width, _ = img.shape
        
        gt_box = np.array([[bbox.left, bbox.top, bbox.right, bbox.bottom]])
        square_gt_box = np.squeeze(convert_to_square(gt_box))
        # 防止越界，同时保持方形
        if square_gt_box[0] < 0:
            square_gt_box[2] -= square_gt_box[0]
            square_gt_box[0] = 0
        if square_gt_box[1] < 0:
            square_gt_box[3] -= square_gt_box[1]
            square_gt_box[1] = 0
        if square_gt_box[2] > image_width:
            square_gt_box[0] -= (square_gt_box[2] - image_width)
            square_gt_box[2] = image_width
        if square_gt_box[3] > image_height:
            square_gt_box[1] -= (square_gt_box[3] - image_height)
            square_gt_box[3] = image_height
            
        gt_box = np.squeeze(gt_box)
        # 计算标准化的关键点坐标
        landmark = np.zeros((5, 2))
        for index, one in enumerate(landmarkGt):
            # (( x - bbox.left)/ width of bounding box, (y - bbox.top)/ height of bounding box
            landmark[index] = ((one[0] - square_gt_box[0]) / (square_gt_box[2] - square_gt_box[0]),
                               (one[1] - square_gt_box[1]) / (square_gt_box[3] - square_gt_box[1]))
        cropped_landmarks.append(landmark.reshape(10))
        
        # 截取目标区域图片
        cropped_object_image = img[square_gt_box[1]:square_gt_box[3] + 1, square_gt_box[0]:square_gt_box[2] + 1]
        cropped_object_image = cv2.resize(cropped_object_image, (size, size))
        cropped_images.append(cropped_object_image)
        
        landmark = np.zeros((5, 2))
        if argument:
            landmark_idx = landmark_idx + 1
            if landmark_idx % 100 == 0:
                sys.stdout.write("\r{}/{} images done ...".format(landmark_idx, len(data)))
                
            # ground truth的坐标、宽和高
            x_truth_left, y_truth_top, x_truth_right, y_truth_bottom = gt_box
            width_truth = x_truth_right - x_truth_left + 1
            height_truth = y_truth_bottom - y_truth_top + 1
            if max(width_truth, height_truth) < 20 or x_truth_left < 0 or y_truth_top < 0:
                continue
            # 随机偏移
            shift_num = 0
            shift_try = 0
            while shift_num < 10 and shift_try < 100:
                bbox_size = npr.randint(int(min(width_truth, height_truth) * 0.8),
                                        np.ceil(1.25 * max(width_truth, height_truth)))
                delta_x = npr.randint(int(-width_truth * 0.2), np.ceil(width_truth * 0.2))
                delta_y = npr.randint(int(-height_truth * 0.2), np.ceil(height_truth * 0.2))
                x_left_shift = int(max(x_truth_left + width_truth / 2 - bbox_size / 2 + delta_x, 0))
                y_top_shift = int(max(y_truth_top + height_truth / 2 - bbox_size / 2 + delta_y, 0))
                x_right_shift = x_left_shift + bbox_size
                y_bottom_shift = y_top_shift + bbox_size
                if x_right_shift > image_width or y_bottom_shift > image_height:
                    shift_try += 1
                    continue
                crop_box = np.array([x_left_shift, y_top_shift, x_right_shift, y_bottom_shift])
                # 计算数据增强后的偏移区域和ground truth的方形校正IoU
                iou = square_IoU(crop_box, np.expand_dims(gt_box, 0))
                if iou > 0.65:
                    shift_num += 1
                    cropped_im = img[y_top_shift:y_bottom_shift + 1, x_left_shift:x_right_shift + 1, :]
                    resized_im = cv2.resize(cropped_im, (size, size))
                    cropped_images.append(resized_im)
                    # 标准化
                    for index, one in enumerate(landmarkGt):
                        landmark[index] = ((one[0] - x_left_shift) / bbox_size, (one[1] - y_top_shift) / bbox_size)
                    cropped_landmarks.append(landmark.reshape(10))
                    
                    # 进行其他类型的数据增强
                    landmark = np.zeros((5, 2))
                    landmark_ = cropped_landmarks[-1].reshape(-1, 2)
                    bbox = BBox([x_left_shift, y_top_shift, x_right_shift, y_bottom_shift])
                    # 镜像
                    if random.choice([0, 1]) > 0:
                        face_flipped, landmark_flipped = flip(resized_im, landmark_)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        cropped_images.append(face_flipped)
                        cropped_landmarks.append(landmark_flipped.reshape(10))
                        
                    # 顺时针旋转
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = \
                            rotate(img, bbox, bbox.reprojectLandmark(landmark_), 5)
                        # landmark_offset
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        cropped_images.append(face_rotated_by_alpha)
                        cropped_landmarks.append(landmark_rotated.reshape(10))
                        # 上下翻转
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        cropped_images.append(face_flipped)
                        cropped_landmarks.append(landmark_flipped.reshape(10))

                    # 逆时针旋转
                    if random.choice([0, 1]) > 0:
                        face_rotated_by_alpha, landmark_rotated = \
                            rotate(img, bbox, bbox.reprojectLandmark(landmark_), -5)
                        landmark_rotated = bbox.projectLandmark(landmark_rotated)
                        face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (size, size))
                        cropped_images.append(face_rotated_by_alpha)
                        cropped_landmarks.append(landmark_rotated.reshape(10))
                        # 上下翻转
                        face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                        face_flipped = cv2.resize(face_flipped, (size, size))
                        cropped_images.append(face_flipped)
                        cropped_landmarks.append(landmark_flipped.reshape(10))
                else:
                    shift_try += 1
                    
        # 保存关键点训练图片及坐标信息
        cropped_images, cropped_landmarks = np.asarray(cropped_images), np.asarray(cropped_landmarks)
        for i in range(len(cropped_images)):
            if np.any(cropped_landmarks[i] < 0):
                continue
            if np.any(cropped_landmarks[i] > 1):
                continue
            
            cv2.imwrite(os.path.join(landmark_dir, "%d.jpg" % image_id), cropped_images[i])
            landmarks = map(str, list(cropped_landmarks[i]))
            landmark_file.write(os.path.join(landmark_dir, "%d.jpg" % image_id) + " -2 " + " ".join(landmarks) + "\n")
            image_id = image_id + 1
    
    landmark_file.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='产生相关网络的关键点训练数据',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--net', dest='net', help='网络类型（PNet, RNet, ONet）',
                        default='PNet', type=str)
    args = parser.parse_args()
    print(args)
    generate_landmark_data(path_config.landmark_train_txt_path, path_config.images_dir, args.net, argument=True)
