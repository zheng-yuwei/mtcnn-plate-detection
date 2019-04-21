# coding:utf-8
import tensorflow as tf
from tensorflow.contrib import slim

import numpy as np
from train_models.MTCNN_config import config

num_keep_radio = 0.7


# 定义 prelu
def prelu(inputs):
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5
    return pos + neg


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    # num_sample*num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def cls_ohem(cls_prob, label):
    """ OHEM分类误差损失函数：正样本、landmark样本预测为1，负样本预测为0，部分样本排除
    :param cls_prob: 预测类别，（batch， 2）
    :param label: 实际类别label，（batch，）
    :return:
    """
    # 训练样本中，正样本 1，负样本 0， 部分样本 -1， landmark样本 -2
    '''
    # 将landmark样本标签-2置为标签1
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label, dtype=tf.float32)
    landmark_indices = tf.where(tf.equal(label, -2), ones_index, zeros_index)
    label += (3 * ones_index * landmark_indices)
    '''
    # 将label中部分样本 -1置为0，则此时样本只有1和0标签
    zeros = tf.zeros_like(label)
    label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)
    label_int = tf.cast(label_filter_invalid, tf.int32)
    
    # 将预测label向量变为一维
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshape = tf.reshape(cls_prob, [num_cls_prob, -1])
    # 计算实际label在一维的预测label向量中的indices： 预测label的起点[0,2,4.....] + 偏移label_int
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    indices_ = tf.range(num_row) * 2 + label_int
    # 获取预测label对应位置的概率，以计算损失
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    loss = -tf.log(label_prob + 1e-10)
    
    # 使标志向量中，正样本和负样本为1，其他样本为0，只计算正负样本损失
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)
    valid_indices = tf.where(label < zeros, zeros, ones)
    loss = loss * valid_indices
    # 应用OHEM
    num_valid = tf.reduce_sum(valid_indices)
    keep_num = tf.cast(num_valid * config.CLS_OHEM_RATIO, dtype=tf.int32)
    loss, _ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)


def bbox_ohem_smooth_L1_loss(bbox_predict, bbox_target, label):
    sigma = tf.constant(1.0)
    threshold = 1.0 / (sigma ** 2)
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    valid_indices = tf.where(label != zeros_index, tf.ones_like(label, dtype=tf.float32), zeros_index)
    abs_error = tf.abs(bbox_predict - bbox_target)
    loss_smaller = 0.5 * ((abs_error * sigma) ** 2)
    loss_larger = abs_error - 0.5 / (sigma ** 2)
    smooth_loss = tf.reduce_sum(tf.where(abs_error < threshold, loss_smaller, loss_larger), axis=1)
    keep_num = tf.cast(tf.reduce_sum(valid_indices) * config.BBOX_OHEM_RATIO, dtype=tf.int32)
    smooth_loss = smooth_loss * valid_indices
    _, k_index = tf.nn.top_k(smooth_loss, k=keep_num)
    smooth_loss_picked = tf.gather(smooth_loss, k_index)
    return tf.reduce_mean(smooth_loss_picked)


# label=1 or label=-1 then do regression
def bbox_ohem(bbox_predict, bbox_target, label):
    """
    :param bbox_predict:
    :param bbox_target:
    :param label: class label
    :return: mean euclidean loss for all the pos and part examples
    """
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label, dtype=tf.float32)
    # 保留正样本和部分样本
    valid_inds = tf.where(tf.equal(tf.abs(label), 1), ones_index, zeros_index)
    # ground truth边框四周留一定余量，不受惩罚
    width = bbox_target[:, 2] - bbox_target[:, 0]
    height = bbox_target[:, 3] - bbox_target[:, 1]
    # width/20>left实际值-left预测值>0，则为0，否则为1
    pos0 = tf.where(tf.greater(bbox_target[:, 0] - bbox_predict[:, 0], 0.0), zeros_index, ones_index)
    pos0 = pos0 + tf.where(tf.greater(width/20.0, bbox_target[:, 0] - bbox_predict[:, 0]), zeros_index, ones_index)
    # height/20>top实际值-top预测值>0，则为0，否则为1
    pos1 = tf.where(tf.greater(bbox_target[:, 1] - bbox_predict[:, 1], 0.0), zeros_index, ones_index)
    pos1 = pos1 + tf.where(tf.greater(height / 20.0, bbox_target[:, 1] - bbox_predict[:, 1]), zeros_index, ones_index)
    # width/20>right预测值-right实际值>0，则为0，否则为1
    pos2 = tf.where(tf.greater(bbox_predict[:, 2] - bbox_target[:, 2], 0.0), zeros_index, ones_index)
    pos2 = pos2 + tf.where(tf.greater(width / 20.0, bbox_predict[:, 2] - bbox_target[:, 2]), zeros_index, ones_index)
    # height/20>bottom实际值-bottom预测值>0，则为0，否则为1
    pos3 = tf.where(tf.greater(bbox_target[:, 3] - bbox_predict[:, 3], 0.0), zeros_index, ones_index)
    pos3 = pos3 + tf.where(tf.greater(height / 20.0, bbox_target[:, 3] - bbox_predict[:, 3]), zeros_index, ones_index)
    flag = tf.stack([pos0, pos1, pos2, pos3], axis=1)
    delta = tf.multiply(bbox_predict - bbox_target, flag)
    
    # 计算均方差损失
    # delta = bbox_predict - bbox_target
    square_error = tf.square(delta)
    square_error = tf.reduce_sum(square_error, axis=1)
    # keep_num scalar
    num_valid = tf.reduce_sum(valid_inds)
    # keep_num = tf.cast(num_valid*config.BBOX_OHEM_RATIO, dtype=tf.int32)
    # count the number of pos and part examples
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    # keep valid index square_error
    square_error = square_error * valid_inds
    # keep top k examples, k equals to the number of positive examples
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    
    return tf.reduce_mean(square_error)


def landmark_ohem(landmark_predict, landmark_target, label):
    """
    :param landmark_predict:
    :param landmark_target:
    :param label:
    :return: mean euclidean loss
    """
    # keep label =-2  then do landmark detection
    ones = tf.ones_like(label, dtype=tf.float32)
    zeros = tf.zeros_like(label, dtype=tf.float32)
    valid_inds = tf.where(tf.equal(label, -2), ones, zeros)
    square_error = tf.square(landmark_predict - landmark_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    # keep_num = tf.cast(num_valid*config.BBOX_OHEM_RATIO, dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error * valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)


def cal_accuracy(cls_prob, label):
    """ 计算精度
    :param cls_prob:
    :param label:
    :return:calculate classification accuracy for pos and neg examples only
    """
    # get the index of maximum value along axis one from cls_prob
    # 0 for negative 1 for positive
    pred = tf.argmax(cls_prob, axis=1)
    label_int = tf.cast(label, tf.int64)
    # return the index of pos and neg examples
    cond = tf.where(tf.greater_equal(label_int, 0))
    picked = tf.squeeze(cond)
    # gather the label of pos and neg examples
    label_picked = tf.gather(label_int, picked)
    predict_picked = tf.gather(pred, picked)
    # calculate the mean value of a vector contains 1 and 0, 1 for correct classification, 0 for incorrect
    # ACC = (TP+FP)/total population
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked, predict_picked), tf.float32))
    return accuracy_op


def _activation_summary(x):
    """
    creates a summary provides histogram of activations
    creates a summary that measures the sparsity of activations
    :param x: Tensor
    :return:
    """
    tensor_name = x.op.name
    print('load summary for : ', tensor_name)
    tf.summary.histogram(tensor_name + '/activations', x)
    # tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


# 构造PNet
def P_Net(inputs, label=None, bbox_target=None, landmark_target=None, training=True):
    # define common param
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        print('PNet模型输入形状：', inputs.get_shape())
        net = slim.conv2d(inputs, 10, kernel_size=[3, 3], stride=1, scope='conv1')
        _activation_summary(net)
        print('conv1层输出形状：', net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool1', padding='SAME')
        _activation_summary(net)
        print('最大pool1层输出形状：', net.get_shape())
        net = slim.conv2d(net, num_outputs=16, kernel_size=[3, 3], stride=1, scope='conv2')
        _activation_summary(net)
        print('conv2层输出形状：', net.get_shape())
        net = slim.conv2d(net, num_outputs=32, kernel_size=[3, 3], stride=1, scope='conv3')
        _activation_summary(net)
        print('conv3层输出形状：', net.get_shape())
        # 类别预测输出：batch * H * W * 2
        conv4_1 = slim.conv2d(net, num_outputs=2, kernel_size=[1, 1], stride=1, scope='conv4_1',
                              activation_fn=tf.nn.softmax)
        print('class层输出形状：', conv4_1.get_shape())
        # bounding boxes预测输出：batch * H * W * 4
        bbox_predict = slim.conv2d(net, num_outputs=4, kernel_size=[1, 1], stride=1, scope='conv4_2',
                                   activation_fn=None)
        _activation_summary(bbox_predict)
        print('bounding box层输出形状：', bbox_predict.get_shape())
        # 5个关键点预测输出：batch * H * W * 10
        landmark_predict = slim.conv2d(net, num_outputs=10, kernel_size=[1, 1], stride=1, scope='conv4_3',
                                       activation_fn=None)
        _activation_summary(landmark_predict)
        print('land mark层输出形状：', landmark_predict.get_shape())
    
        # add projectors for visualization
    
        if training:
            # OHEM计算分类(batch * 2)损失
            cls_prob = tf.squeeze(conv4_1, [1, 2], name='cls_prob')
            cls_loss = cls_ohem(cls_prob, label)
            # OHEM（其实没用）均方误差计算bounding box损失（batch * 4）
            bbox_predict = tf.squeeze(bbox_predict, [1, 2], name='bbox_predict')
            bbox_loss = bbox_ohem(bbox_predict, bbox_target, label)
            # OHEM（其实没用）均方误差计算5个关键点损失（batch * 10）
            landmark_predict = tf.squeeze(landmark_predict, [1, 2], name="landmark_predict")
            landmark_loss = landmark_ohem(landmark_predict, landmark_target, label)
            # 计算精度
            accuracy = cal_accuracy(cls_prob, label)
            # 权重罚项损失
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
        else:
            # 测试时，batch = 1
            cls_pro_test = tf.squeeze(conv4_1, axis=0)
            bbox_predict_test = tf.squeeze(bbox_predict, axis=0)
            landmark_predict_test = tf.squeeze(landmark_predict, axis=0)
            return cls_pro_test, bbox_predict_test, landmark_predict_test


# 构建RNet
def R_Net(inputs, label=None, bbox_target=None, landmark_target=None, training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        print('RNet模型输入形状：', inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=28, kernel_size=[3, 3], stride=1, scope="conv1")
        print('conv1层输出形状：', net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print('pool1层输出形状：', net.get_shape())
        net = slim.conv2d(net, num_outputs=48, kernel_size=[3, 3], stride=1, scope="conv2")
        print('conv2层输出形状：', net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        print('pool2层输出形状：', net.get_shape())
        net = slim.conv2d(net, num_outputs=64, kernel_size=[2, 2], stride=1, scope="conv3")
        print('conv3层输出形状：', net.get_shape())
        fc_flatten = slim.flatten(net)
        print('fc_flatten层输出形状：', fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs=128, scope="fc1")
        print('fc1层输出形状：', fc1.get_shape())
        # 分别获取分类、bounding box、land mark的预测值
        cls_prob = slim.fully_connected(fc1, num_outputs=2, scope="cls_fc", activation_fn=tf.nn.softmax)
        print('cls_fc层输出形状：', cls_prob.get_shape())
        bbox_predict = slim.fully_connected(fc1, num_outputs=4, scope="bbox_fc", activation_fn=None)
        print('bbox_fc层输出形状：', bbox_predict.get_shape())
        landmark_predict = slim.fully_connected(fc1, num_outputs=10, scope="landmark_fc", activation_fn=None)
        print('landmark_fc层输出形状：', landmark_predict.get_shape())
        
        if training:
            cls_loss = cls_ohem(cls_prob, label)
            bbox_loss = bbox_ohem(bbox_predict, bbox_target, label)
            accuracy = cal_accuracy(cls_prob, label)
            landmark_loss = landmark_ohem(landmark_predict, landmark_target, label)
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
        else:
            return cls_prob, bbox_predict, landmark_predict


# 构建ONet
def O_Net(inputs, label=None, bbox_target=None, landmark_target=None, training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=prelu,
                        weights_initializer=slim.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        padding='valid'):
        print('ONet模型输入形状：', inputs.get_shape())
        net = slim.conv2d(inputs, num_outputs=32, kernel_size=[3, 3], stride=1, scope="conv1")
        print('conv1层输出形状：', net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool1", padding='SAME')
        print('pool1层输出形状：', net.get_shape())
        net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], stride=1, scope="conv2")
        print('conv2层输出形状：', net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope="pool2")
        print('pool2层输出形状：', net.get_shape())
        net = slim.conv2d(net, num_outputs=64, kernel_size=[3, 3], stride=1, scope="conv3")
        print('conv3层输出形状：', net.get_shape())
        net = slim.max_pool2d(net, kernel_size=[2, 2], stride=2, scope="pool3", padding='SAME')
        print('pool3层输出形状：', net.get_shape())
        net = slim.conv2d(net, num_outputs=128, kernel_size=[2, 2], stride=1, scope="conv4")
        print('conv4层输出形状：', net.get_shape())
        fc_flatten = slim.flatten(net)
        print('fc_flatten层输出形状：', fc_flatten.get_shape())
        fc1 = slim.fully_connected(fc_flatten, num_outputs=256, scope="fc1")
        print('fc1层输出形状：', fc1.get_shape())
        # 分别获取分类、bounding box、land mark的预测值
        cls_prob = slim.fully_connected(fc1, num_outputs=2, scope="cls_fc", activation_fn=tf.nn.softmax)
        print('cls_fc层输出形状：', cls_prob.get_shape())
        bbox_predict = slim.fully_connected(fc1, num_outputs=4, scope="bbox_fc", activation_fn=None)
        print('bbox_fc层输出形状：', bbox_predict.get_shape())
        landmark_predict = slim.fully_connected(fc1, num_outputs=10, scope="landmark_fc", activation_fn=None)
        print('landmark_fc层输出形状：', landmark_predict.get_shape())
        
        if training:
            cls_loss = cls_ohem(cls_prob, label)
            bbox_loss = bbox_ohem(bbox_predict, bbox_target, label)
            accuracy = cal_accuracy(cls_prob, label)
            landmark_loss = landmark_ohem(landmark_predict, landmark_target, label)
            L2_loss = tf.add_n(tf.losses.get_regularization_losses())
            return cls_loss, bbox_loss, landmark_loss, L2_loss, accuracy
        else:
            return cls_prob, bbox_predict, landmark_predict
