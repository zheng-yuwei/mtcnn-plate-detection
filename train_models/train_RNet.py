# -*- coding: utf-8 -*-
"""
Created on 2019/4/11
File pnet_test
@author:ZhengYuwei
功能：训练PNet网络
"""
from train_models.mtcnn_model import R_Net
from train_models.train import train
from prepare_data.path_configure import PathConfiguration


def train_RNet(tfrecord_path, model_save_path, max_epoch, display, lr):
    """ train RNet
    :param tfrecord_path: tfrecord训练数据路径
    :param model_save_path: 模型保存路径
    :param max_epoch: 最大训练epoch
    :param display: 日志打印
    :param lr: learning rate
    :return:
    """
    net_factory = R_Net
    train(net_factory, model_save_path, max_epoch, tfrecord_path, display=display, base_lr=lr)


if __name__ == '__main__':
    path_config = PathConfiguration().config
    # 训练用的tfrecord数据路径
    tfrecord_folder = [path_config.rnet_merge_txt_path,
                       path_config.rnet_pos_tfrecord_path_shuffle,
                       path_config.rnet_part_tfrecord_path_shuffle,
                       path_config.rnet_neg_tfrecord_path_shuffle,
                       path_config.rnet_landmark_tfrecord_path_shuffle]
    # 模型保存路径
    rnet_model_folder = path_config.rnet_landmark_model_path
    # max_epoch = 22
    train_RNet(tfrecord_folder, rnet_model_folder, max_epoch=40, display=100, lr=0.001)
