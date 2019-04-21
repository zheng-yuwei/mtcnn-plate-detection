# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import argparse
from prepare_data.path_configure import PathConfiguration

path_config = PathConfiguration().config
data_dir = path_config.root_path


def generate_merge_data(net):
    """ 合并前面生成的正样本、负样本、部分样本，以及关键点样本数据
    :param net: 训练样本所属的网络类型
    :return:
    """
    if net == 'PNet':
        pos_file_path = path_config.pnet_pos_txt_path
        part_file_path = path_config.pnet_part_txt_path
        neg_file_path = path_config.pnet_neg_txt_path
        landmark_file_path = path_config.pnet_landmark_txt_path
        merge_root_dir = path_config.pnet_merge_root_path
        merge_txt_path = path_config.pnet_merge_txt_path
    elif net == 'RNet':
        pos_file_path = path_config.rnet_pos_txt_path
        part_file_path = path_config.rnet_part_txt_path
        neg_file_path = path_config.rnet_neg_txt_path
        landmark_file_path = path_config.rnet_landmark_txt_path
        merge_root_dir = path_config.rnet_merge_root_path
        merge_txt_path = path_config.rnet_merge_txt_path
    elif net == 'ONet':
        pos_file_path = path_config.onet_pos_txt_path
        part_file_path = path_config.onet_part_txt_path
        neg_file_path = path_config.onet_neg_txt_path
        landmark_file_path = path_config.onet_landmark_txt_path
        merge_root_dir = path_config.onet_merge_root_path
        merge_txt_path = path_config.onet_merge_txt_path
    else:
        raise ValueError('网络类型(--net)错误！')
        
    with open(pos_file_path, 'r') as pos_file:
        pos = pos_file.readlines()
    with open(neg_file_path, 'r') as neg_file:
        neg = neg_file.readlines()
    with open(part_file_path, 'r') as part_file:
        part = part_file.readlines()
    with open(landmark_file_path, 'r') as landmark_file:
        landmark = landmark_file.readlines()
    
    if not os.path.exists(merge_root_dir):
        os.makedirs(merge_root_dir)
        
    with open(merge_txt_path, "w") as merge_file:
        nums = [len(neg), len(pos), len(part), len(landmark)]
        # PNet要平衡数据
        base_num = None
        if net == 'PNet':
            ratio = [3.0, 1.0, 1.0, 1.0]
            # 重采样以平衡数据
            base_num = max([np.ceil(nums[i] / ratio[i]) for i in range(3)])
            # 欠采样以平衡数据
            # base_num = min([np.ceil(nums[i] / ratio[i]) for i in range(3)])
            print('各类样本数据：{}, 基本数据量：{}'.format(nums, base_num))
        
            keep = list()
            for i in range(3):
                if nums[i] < ratio[i] * base_num:
                    keep.append(np.append(np.array(range(nums[i])),
                                          np.random.choice(nums[i], size=int(ratio[i] * base_num - nums[i]),
                                                           replace=True)))
                elif nums[i] == ratio[i] * base_num:
                    keep.append(np.array(range(nums[i])))
                else:
                    keep.append(np.random.choice(nums[i], size=int(ratio[i] * base_num), replace=False))
            keep.append(np.array(range(nums[3])))
        # RNet和ONet在训练时，会进行数据平衡
        else:
            keep = [np.array(range(nums[i])) for i in range(4)]
        
        # 随机扰乱数据
        for i in range(4):
            random.shuffle(keep[i])
        neg_keep, pos_keep, part_keep, landmark_keep = keep
        print('各类样本数据：{}, 基本数据量：{}'.format(
            [len(neg_keep), len(pos_keep), len(part_keep), len(landmark_keep)], base_num))
        
        # 合并到一个数据文件中
        for i in pos_keep:
            merge_file.write(pos[i])
        for i in neg_keep:
            merge_file.write(neg[i])
        for i in part_keep:
            merge_file.write(part[i])
        for i in landmark_keep:
            merge_file.write(landmark[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='产生相关网络的关键点训练数据',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--net', dest='net', help='网络类型（PNet, RNet, ONet）',
                        default='PNet', type=str)
    args = parser.parse_args()
    generate_merge_data(args.net)
