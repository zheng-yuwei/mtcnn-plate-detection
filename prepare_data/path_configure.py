# -*- coding: utf-8 -*-
"""
Created on 2019/4/4
File config
@author:ZhengYuwei
"""
import os
from easydict import EasyDict


class PathConfiguration(object):
    """
    配置信息类（单例模式）
    """
    
    def __new__(cls, *args, **kw):
        if not hasattr(PathConfiguration, "_instance"):
            PathConfiguration._instance = super(PathConfiguration, cls).__new__(cls, *args, **kw)
        return PathConfiguration._instance
    
    def __init__(self):
        path_suffix = {
            # 数据集根目录
            'root_path': '/home/data_160/data3/smart_home/xiongxin-yuwei/PlateNum_HeChang/mtcnn/',  # '../../DATA'
            # 输入文件路径
            'images_dir': 'data/images',  # 图片所在路径
            'point2_train_txt_path': 'data/bounding_boxes_train.txt',  # bounding boxes训练数据
            'landmark_train_txt_path': 'data/landmarks_train.txt',  # 关键点训练数据
            'point2_val_txt_path': 'data/bounding_boxes_validate.txt',  # bounding boxes验证数据
            'landmark_val_txt_path': 'data/landmarks_validate.txt',  # 关键点验证数据
            'point2_test_txt_path': 'data/bounding_boxes_test.txt',  # bounding boxes测试数据
            'landmark_test_txt_path': 'data/landmarks_test.txt',  # 关键点测试数据
            'point2_all_txt_path': 'data/bounding_boxes_all.txt',  # bounding boxes全量数据
            'landmark_all_txt_path': 'data/landmarks_all.txt',  # 关键点全量数据
            
            # PNet相关数据的路径
            'pnet_dir': '12',  # PNet数据根目录
            'pnet_pos_dir': '12/positive',       # 截取的正样本图片路径
            'pnet_part_dir': '12/part',          # 截取的部分样本图片路径
            'pnet_neg_dir': '12/negative',       # 截取的负样本图片路径
            'pnet_landmark_dir': '12/landmark',  # 截取的关键点样本图片路径
            'pnet_pos_txt_path': '12/pnet_pos.txt',            # 记录截取的正样本图片信息的txt文件路径
            'pnet_part_txt_path': '12/pnet_part.txt',          # 记录截取的部分样本图片信息的txt文件路径
            'pnet_neg_txt_path': '12/pnet_neg.txt',            # 记录截取的负样本图片信息的txt文件路径
            'pnet_landmark_txt_path': '12/pnet_landmark.txt',  # 记录截取的关键点样本图片信息的txt文件路径
            'pnet_merge_root_path': 'merge/PNet',                             # 合并不同样本数据的根目录
            'pnet_merge_txt_path': 'merge/PNet/merge_data.txt',      # 合并不同样本信息数据的txt文件
            'pnet_tfrecord_path': 'merge/PNet/merge_data.tfrecord',  # 合并不同样本图片的tfrecord文件
            'pnet_tfrecord_path_shuffle': 'merge/PNet/merge_data.tfrecord_shuffle',  # 随机扰乱
            'pnet_landmark_model_path': 'MTCNN_model/PNet_model/PNet',  # 训练后的PNet模型参数保存路径
            'pnet_log_path': 'logs/PNet',  # PNet训练日志路径
            
            # RNet相关数据的路径
            'rnet_dir': '24',  # RNet数据根目录
            # PNet网络的困难样本bounding box的pkl文件，需要输入RNet进一步refine
            'rnet_save_hard_path': '24/RNet',          # 经训练后的PNet检测得到的候选框数据
            'rnet_pos_txt_path': '24/rnet_pos.txt',    # 根据候选框的iou，判定为正样本的候选框数据
            'rnet_part_txt_path': '24/rnet_neg.txt',   # 根据候选框的iou，判定为负样本的候选框数据
            'rnet_neg_txt_path': '24/rnet_part.txt',   # 根据候选框的iou，判定为部分样本的候选框数据
            'rnet_pos_dir': '24/positive',           # 根据候选框的iou，判定为正样本的图片数据目录
            'rnet_part_dir': '24/part',              # 根据候选框的iou，判定为部分样本的图片数据目录
            'rnet_neg_dir': '24/negative',           # 根据候选框的iou，判定为负样本的图片数据目录
            'rnet_landmark_txt_path': '24/rnet_landmark.txt',  # 截取的关键点样本坐标数据
            'rnet_landmark_dir': '24/landmark',                # 截取的关键点样本图片
            'rnet_merge_root_path': 'merge/RNet',   # 将图片数据转为tfrecord文件
            'rnet_merge_txt_path': 'merge/RNet/merge_data.txt',  # 合并不同样本信息数据的txt文件
            'rnet_pos_tfrecord_path': 'merge/RNet/positive.tfrecord',       # 正样本图片的tfrecord文件
            'rnet_part_tfrecord_path': 'merge/RNet/part.tfrecord',          # 部分样本图片的tfrecord文件
            'rnet_neg_tfrecord_path': 'merge/RNet/negative.tfrecord',       # 负样本图片的tfrecord文件
            'rnet_landmark_tfrecord_path': 'merge/RNet/landmark.tfrecord',  # 关键点样本图片的tfrecord文件
            'rnet_pos_tfrecord_path_shuffle': 'merge/RNet/positive.tfrecord_shuffle',       # 扰乱
            'rnet_part_tfrecord_path_shuffle': 'merge/RNet/part.tfrecord_shuffle',
            'rnet_neg_tfrecord_path_shuffle': 'merge/RNet/negative.tfrecord_shuffle',
            'rnet_landmark_tfrecord_path_shuffle': 'merge/RNet/landmark.tfrecord_shuffle',
            'rnet_landmark_model_path': 'MTCNN_model/RNet_model/RNet',  # 训练后的RNet模型参数保存路径
            'rnet_log_path': 'logs/RNet',  # RNet训练日志路径
            
            # ONet相关数据路径
            'onet_dir': '48',  # ONet数据根目录
            'onet_save_hard_path': '48/ONet',    # 经训练后的RNet检测得到的候选框数据
            'onet_pos_txt_path': '48/onet_pos.txt',    # 根据候选框的iou，判定为正样本的候选框数据
            'onet_part_txt_path': '48/onet_neg.txt',   # 根据候选框的iou，判定为负样本的候选框数据
            'onet_neg_txt_path': '48/onet_part.txt',   # 根据候选框的iou，判定为部分样本的候选框数据
            'onet_pos_dir': '48/positive',          # 根据候选框的iou，判定为正样本的图片数据目录
            'onet_part_dir': '48/part',             # 根据候选框的iou，判定为正样本的图片数据目录
            'onet_neg_dir': '48/negative',          # 根据候选框的iou，判定为负样本的图片数据目录
            'onet_landmark_txt_path': '48/onet_landmark.txt',  # 截取的关键点样本坐标数据
            'onet_landmark_dir': '48/landmark',                # 截取的关键点样本图片
            'onet_merge_root_path': 'merge/ONet',  # 将图片数据转为tfrecord文件
            'onet_merge_txt_path': 'merge/ONet/merge_data.txt',  # 合并不同样本信息数据的txt文件
            'onet_pos_tfrecord_path': 'merge/ONet/positive.tfrecord',       # 正样本图片的tfrecord文件
            'onet_part_tfrecord_path': 'merge/ONet/part.tfrecord',          # 部分样本图片的tfrecord文件
            'onet_neg_tfrecord_path': 'merge/ONet/negative.tfrecord',       # 负样本图片的tfrecord文件
            'onet_landmark_tfrecord_path': 'merge/ONet/landmark.tfrecord',  # 关键点样本图片的tfrecord文件
            'onet_pos_tfrecord_path_shuffle': 'merge/ONet/positive.tfrecord_shuffle',  # 扰乱
            'onet_part_tfrecord_path_shuffle': 'merge/ONet/part.tfrecord_shuffle',
            'onet_neg_tfrecord_path_shuffle': 'merge/ONet/negative.tfrecord_shuffle',
            'onet_landmark_tfrecord_path_shuffle': 'merge/ONet/landmark.tfrecord_shuffle',
            'onet_landmark_model_path': 'MTCNN_model/ONet_model/ONet',  # 训练后的ONet模型参数保存路径
            'onet_log_path': 'logs/ONet',  # ONet训练日志路径
            
        }
        for key in path_suffix.keys():
            if key == 'root_path':
                continue
            path_suffix[key] = os.path.join(path_suffix['root_path'], path_suffix.get(key))
        
        # 待检测目标的最小高，宽
        path_suffix['min_height_size'] = 20
        path_suffix['min_width_size'] = int(path_suffix['min_height_size'] * 1.5)
        # 待检测目标的宽高比
        x_y_proportion = 2.7
        # PNet网络输入的高和宽的size
        path_suffix['pnet_height_size'] = 12
        path_suffix['pnet_width_size'] = int(path_suffix['pnet_height_size'] * x_y_proportion)
        # RNet网络输入的高和宽的size
        path_suffix['rnet_height_size'] = 24
        path_suffix['rnet_width_size'] = int(path_suffix['rnet_height_size'] * x_y_proportion)
        # ONet网络输入的高和宽的size
        path_suffix['onet_height_size'] = 48
        path_suffix['onet_width_size'] = int(path_suffix['onet_height_size'] * x_y_proportion)
        self.config = EasyDict(path_suffix)


