#!/usr/bin/env bash
# 先到prepare_data文件夹目录下
# 将包加入模块搜索路径
export PYTHONPATH='..:$PYTHONPATH'
# 过滤目标太小的图片数据
python filter_acceptable_images.py
## PNet
python gen_12net_data.py  # 产生pnet的 正样本、负样本、部分样本 训练数据
# 限制 长宽>=20 !!!
python gen_landmark.py --net=PNet  # generate training data(Face Landmark Detection Part)
python gen_merge_data.py --net=PNet  # merge two parts of training data
python gen_tfrecords.py --net=PNet  # generate tfrecord for PNet
# 产生数据集比例：neg: pos: part: landmark = 3:1:1:1
python train_PNet.py  # 到train_models路径下，training PNet ...

## RNet
python gen_hard_example.py --test_mode=PNet --epoch 14 20 40  # generate training data(Face Detection Part)
python gen_landmark.py --net=RNet  # generate training data(Face Landmark Detection Part)
python gen_merge_data.py --net=RNet  # merge two parts of training data
# run this script four times to generate tfrecords of neg,pos,part and landmark respectively
python gen_tfrecords.py --net=RNet  # generate tfrecord for RNe
# 产生数据集比例：pos: part: landmark: neg = 1:1:1:3
python train_RNet.py  # training RNet ...

## ONet
python gen_hard_example.py --test_mode=RNet --epoch 14 20 40  # generate training data(Face Detection Part)
python gen_landmark.py --net=ONet  # generate training data(Face Landmark Detection Part)
python gen_merge_data.py --net=ONet  # merge two parts of training data
# run this script four times to generate tfrecords of neg,pos,part and landmark respectively
python gen_tfrecords.py --net=ONet  # generate tfrecords
python train_ONet.py  # training ONet ...


# 进行测试
python pnet_test.py  # --image_path=
