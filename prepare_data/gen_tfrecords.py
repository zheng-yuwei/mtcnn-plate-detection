# coding:utf-8
import argparse
import sys
import random
import tensorflow as tf

from prepare_data.tfrecord_utils import _process_image_withoutcoder, _convert_to_example_simple
from prepare_data.path_configure import PathConfiguration

path_config = PathConfiguration().config


def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    """ 从图片和annotations中加载数据，添加到TFRecord文件
    :param filename: 图片文件名称
    :param image_example: 包含样本信息的数据结构
    :param tfrecord_writer: The TFRecord writer to use for writing.
    """
    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def run(net, shuffling=False):
    """Runs the conversion operation.
    :param net: 网络类型
    :param shuffling: 是否随机扰乱
    """
    # tfrecord文件路径，信息txt文件路径
    if net == 'PNet':
        if shuffling:
            tf_file_paths = [path_config.pnet_tfrecord_path_shuffle]
        else:
            tf_file_paths = [path_config.pnet_tfrecord_path]
        data_txt_paths = [path_config.pnet_merge_txt_path]
    elif net == 'RNet':
        if shuffling:
            tf_file_paths = [path_config.rnet_pos_tfrecord_path_shuffle,
                             path_config.rnet_part_tfrecord_path_shuffle,
                             path_config.rnet_neg_tfrecord_path_shuffle,
                             path_config.rnet_landmark_tfrecord_path_shuffle]
        else:
            tf_file_paths = [path_config.rnet_pos_tfrecord_path,
                             path_config.rnet_part_tfrecord_path,
                             path_config.rnet_neg_tfrecord_path,
                             path_config.rnet_landmark_tfrecord_path]
        data_txt_paths = [path_config.rnet_pos_txt_path,
                          path_config.rnet_part_txt_path,
                          path_config.rnet_neg_txt_path,
                          path_config.rnet_landmark_txt_path]
    elif net == 'ONet':
        if shuffling:
            tf_file_paths = [path_config.onet_pos_tfrecord_path_shuffle,
                             path_config.onet_part_tfrecord_path_shuffle,
                             path_config.onet_neg_tfrecord_path_shuffle,
                             path_config.onet_landmark_tfrecord_path_shuffle]
        else:
            tf_file_paths = [path_config.onet_pos_tfrecord_path,
                             path_config.onet_part_tfrecord_path,
                             path_config.onet_neg_tfrecord_path,
                             path_config.onet_landmark_tfrecord_path]
        data_txt_paths = [path_config.onet_pos_txt_path,
                          path_config.onet_part_txt_path,
                          path_config.onet_neg_txt_path,
                          path_config.onet_landmark_txt_path]
    else:
        raise ValueError('网络类型(--net)错误!')
    
    for i in range(len(tf_file_paths)):
        if tf.gfile.Exists(tf_file_paths[i]):
            print('TFRecord数据集({})已经存在，不再生成...'.format(tf_file_paths[i]))
            continue
        
        # 读取信息txt文件
        dataset = get_dataset(data_txt_paths[i])
        if shuffling:
            random.shuffle(dataset)
    
        with tf.python_io.TFRecordWriter(tf_file_paths[i]) as tfrecord_writer:
            for j, image_example in enumerate(dataset):
                if (j + 1) % 10000 == 0:
                    sys.stdout.write('\r>> %d-%d/%d images has been converted' % (i, j + 1, len(dataset)))
                sys.stdout.flush()
                filename = image_example['filename']
                _add_to_tfrecord(filename, image_example, tfrecord_writer)
    
    print('\nFinished converting the MTCNN dataset!')


def get_dataset(data_txt_path):
    with open(data_txt_path, 'r') as data_file:
        dataset = []
        for line in data_file.readlines():
            info = line.strip().split(' ')
            data_example = dict()
            bbox = dict()
            data_example['filename'] = info[0]
            data_example['label'] = int(info[1])
            bbox['xmin'] = 0
            bbox['ymin'] = 0
            bbox['xmax'] = 0
            bbox['ymax'] = 0
            bbox['xlefteye'] = 0
            bbox['ylefteye'] = 0
            bbox['xrighteye'] = 0
            bbox['yrighteye'] = 0
            bbox['xnose'] = 0
            bbox['ynose'] = 0
            bbox['xleftmouth'] = 0
            bbox['yleftmouth'] = 0
            bbox['xrightmouth'] = 0
            bbox['yrightmouth'] = 0
            if len(info) == 6:
                bbox['xmin'] = float(info[2])
                bbox['ymin'] = float(info[3])
                bbox['xmax'] = float(info[4])
                bbox['ymax'] = float(info[5])
            if len(info) == 12:
                bbox['xlefteye'] = float(info[2])
                bbox['ylefteye'] = float(info[3])
                bbox['xrighteye'] = float(info[4])
                bbox['yrighteye'] = float(info[5])
                bbox['xnose'] = float(info[6])
                bbox['ynose'] = float(info[7])
                bbox['xleftmouth'] = float(info[8])
                bbox['yleftmouth'] = float(info[9])
                bbox['xrightmouth'] = float(info[10])
                bbox['yrightmouth'] = float(info[11])
            
            data_example['bbox'] = bbox
            dataset.append(data_example)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='产生相关网络的关键点训练数据',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--net', dest='net', help='网络类型（PNet, RNet, ONet）',
                        default='PNet', type=str)
    args = parser.parse_args()
    run(args.net, shuffling=True)
