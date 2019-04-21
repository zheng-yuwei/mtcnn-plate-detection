# coding:utf-8
from train_models.mtcnn_model import O_Net
from train_models.train import train
from prepare_data.path_configure import PathConfiguration


def train_ONet(tfrecord_path, model_save_path, max_epoch, display, lr):
    """ train ONet
    :param tfrecord_path: tfrecord训练数据路径
    :param model_save_path: 模型保存路径
    :param max_epoch: 最大训练epoch
    :param display: 日志打印
    :param lr: learning rate
    :return:
    """
    net_factory = O_Net
    train(net_factory, model_save_path, max_epoch, tfrecord_path, display=display, base_lr=lr)


if __name__ == '__main__':
    path_config = PathConfiguration().config
    # 训练用的tfrecord数据路径
    tfrecord_folder = [path_config.onet_merge_txt_path,
                       path_config.onet_pos_tfrecord_path_shuffle,
                       path_config.onet_part_tfrecord_path_shuffle,
                       path_config.onet_neg_tfrecord_path_shuffle,
                       path_config.onet_landmark_tfrecord_path_shuffle]
    # 模型保存路径
    rnet_model_folder = path_config.onet_landmark_model_path
    # max_epoch = 22
    train_ONet(tfrecord_folder, rnet_model_folder, max_epoch=50, display=100, lr=0.001)
