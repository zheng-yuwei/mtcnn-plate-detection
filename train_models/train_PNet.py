# -*- coding: utf-8 -*-
"""
Created on 2019/4/11
File pnet_test
@author:ZhengYuwei
功能：训练PNet网络
"""
from train_models.mtcnn_model import P_Net
from train_models.train import train
from prepare_data.path_configure import PathConfiguration
import os


def train_PNet(tfrecord_path, model_save_path, max_epoch, display, lr):
    """ train PNet
    :param tfrecord_path: tfrecord训练数据路径
    :param model_save_path: 模型保存路径
    :param max_epoch: 最大训练epoch
    :param display: 日志打印
    :param lr: learning rate
    :return:
    """
    net_factory = P_Net
    train(net_factory, model_save_path, max_epoch, tfrecord_path, display=display, base_lr=lr)


if __name__ == '__main__':
    path_config = PathConfiguration().config
    # tfrecord训练数据路径
    tfrecord_folder = [path_config.pnet_merge_txt_path, path_config.pnet_tfrecord_path_shuffle]
    # PNet模型参数保存路径
    pnet_model_folder = path_config.pnet_landmark_model_path
    
    if not os.path.exists(os.path.dirname(pnet_model_folder)):
        os.makedirs(os.path.dirname(pnet_model_folder))
    
    train_PNet(tfrecord_folder, pnet_model_folder, max_epoch=50, display=100, lr=0.001)
