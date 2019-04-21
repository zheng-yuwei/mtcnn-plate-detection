# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import argparse
import pickle as pickle
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from Detection.MtcnnDetector import MtcnnDetector
from prepare_data.utils import IoU, convert_to_square
from prepare_data.path_configure import PathConfiguration
from train_models.MTCNN_config import config

path_config = PathConfiguration().config


def read_bboxes_data(file_path, images_folder):
    """ 从bounding box的txt文件中读取信息
    :param file_path: 保存bounding box信息的txt文件路径
    :param images_folder: 图片根目录
    :return: 图片路径及bounding box信息
    {'images'：路径列表，'bboxes'：对应的bounding boxes}
    """
    images_path = []
    bboxes = []
    with open(file_path, 'r') as bboxes_file:
        for line in bboxes_file:
            line = line.strip().split(' ')
            images_path.append(os.path.join(images_folder, line[0]))
            bboxes.append([[float(_) for _ in line[1:5]]])
    bboxes_data = {'images': images_path, 'bboxes': bboxes}
    return bboxes_data


def save_hard_example(data, test_mode, save_path):
    """ 对模型测试的结果根据预测框和ground truth的IoU进行划分，用于训练下一个网络的困难数据集
    :param data: 模型测试的图片信息数据
    :param test_mode: 测试的网络模型，（PNet，RNet）
    :param save_path: 测试的模型pickle结果保存的路径
    :return:
    """
    im_idx_list = data['images']
    gt_boxes_list = data['bboxes']
    num_of_images = len(im_idx_list)
    print("共需处理图片数：", num_of_images)
    
    # 不同样本图片保存路径
    if test_mode == 'PNet':
        pos_label_file = path_config.rnet_pos_txt_path
        part_label_file = path_config.rnet_part_txt_path
        neg_label_file = path_config.rnet_neg_txt_path
    elif test_mode == 'RNet':
        pos_label_file = path_config.onet_pos_txt_path
        part_label_file = path_config.onet_part_txt_path
        neg_label_file = path_config.onet_neg_txt_path
    else:
        raise ValueError('网络类型(--test_mode)错误！')
    
    pos_file = open(pos_label_file, 'w')
    part_file = open(part_label_file, 'w')
    neg_file = open(neg_label_file, 'w')
    # 读取检测结果pickle数据
    det_boxes = pickle.load(open(os.path.join(save_path, 'detections.pkl'), 'rb'))
    assert len(det_boxes) == num_of_images, "incorrect detections or ground truths"
    
    # 负样本，正样本，部分样本的图片数量，作为文件名
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0  # 已处理图片
    no_pos_image_num = 0  # 没有产生正样本的累积图片数量
    old_p_idx = -1  # 上一张图片的正样本总数
    for im_idx, actual_detections, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        # 当前正样本总数与上一张图片的正样本总数相等，说明当前图片没有产生正样本
        if old_p_idx == p_idx:
            no_pos_image_num += 1
        else:
            old_p_idx = p_idx
        if (image_done + 1) % 100 == 0:
            print("生成进度：{}/{}".format(image_done + 1, num_of_images))
            print("neg:{}, pos:{}, part:{}, no pos image:{}".format(n_idx, p_idx, d_idx, no_pos_image_num))
        image_done += 1
        
        if actual_detections.shape[0] == 0:
            continue
        # 给每个检测框划分为对应的训练样本：IoU<0.3为负样本，0.4~0.65为部分样本，>0.65为正样本
        img = cv2.imread(im_idx)
        # 将检测结果转为方形，因为下一个网络输入为方形输入
        squared_detections = convert_to_square(actual_detections)
        squared_detections[:, 0:4] = np.round(squared_detections[:, 0:4])
        for index, box in enumerate(squared_detections):
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1
            
            # 忽略小图或越界的
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue
            
            # 计算实际检测框和ground truth检测框的IoU，但crop的图片是方形后的区域
            iou = IoU(actual_detections[index], gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            
            # 保存negative样本(IoU<0.3)，并写label文件
            if np.max(iou) < 0.3:
                save_file = os.path.join(neg_dir, "%s.jpg" % n_idx)
                neg_file.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
            # 保存part样本(0.65>IoU>0.4)或positive样本(IoU>0.65)，并写label文件
            else:
                idx = np.argmax(iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt
                
                # 计算bounding box回归量，作为训练样本
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)
                
                if np.max(iou) >= 0.65:
                    save_file = os.path.join(pos_dir, "%s.jpg" % p_idx)
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif np.max(iou) >= 0.4:
                    save_file = os.path.join(part_dir, "%s.jpg" % d_idx)
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    neg_file.close()
    part_file.close()
    pos_file.close()


def t_net(prefix, epoch, batch_size, test_mode, thresh, min_face_size=20,
          stride=2, slide_window=False):
    detectors = [None, None, None]
    # 生成指定模型的困难样本
    print("Test model: ", test_mode)
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    # PNet模型
    print(model_path[0])
    if slide_window:
        p_net = Detector(P_Net, 12, batch_size[0], model_path[0])
    else:
        p_net = FcnDetector(P_Net, model_path[0])
    detectors[0] = p_net
    
    # RNet模型
    if test_mode in ["RNet", "ONet"]:
        print("=================   {}   =================".format(test_mode))
        r_net = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = r_net
    
    # ONet模型，这个模式下生成的样本主要用来观察，而不是训练
    if test_mode == "ONet":
        print("==================   {}   ================".format(test_mode))
        o_net = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = o_net
    
    # 读取bounding box的ground truth及图片，type:dict，include key 'images' and 'bboxes'
    data = read_bboxes_data(path_config.point2_train_txt_path, path_config.images_dir)
    
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)
    
    print('加载原始图片数据，以进行检测及生成困难样本...')
    test_data = TestLoader(data['images'])
    print('加载完成， 开始检测...')
    detections, _ = mtcnn_detector.detect_images(test_data)
    print('检测完成！')
    
    # 保存检测结果
    if test_mode == "PNet":
        save_path = path_config.rnet_save_hard_path
    elif test_mode == "RNet":
        save_path = path_config.onet_save_hard_path
    else:
        raise ValueError('网络类型(--test_mode)错误！')
    print('保存检测结果的路径为:', save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    save_file = os.path.join(save_path, "detections.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(detections, f, 1)
    print("%s测试完成，开始生成困难样本..." % test_mode)
    save_hard_example(data, test_mode, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                        default='ONet', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=[path_config.pnet_landmark_model_path,
                                 path_config.rnet_landmark_model_path,
                                 path_config.onet_landmark_model_path],
                        type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[14, 40, 50], type=int)  # 40, 14, 16
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.7, 0.7, 0.7], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=20, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    # parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',default=0, type=int)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args


# --test_mode='PNet'
# --test_mode='RNet'
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = config.VISIBLE_GPU
    args = parse_args()
    print('Called with argument:')
    print(args)
    
    # 先是RNet，然后ONet
    if args.test_mode == 'PNet':
        net = 'RNet'
    elif args.test_mode == 'RNet':
        net = 'ONet'
    else:
        print('请输入生成困难样本的网络！')
        raise ValueError('--test_mode')
    
    if net == "RNet":
        image_size = 24
        pos_dir = path_config.rnet_pos_dir
        part_dir = path_config.rnet_part_dir
        neg_dir = path_config.rnet_neg_dir
    if net == "ONet":
        image_size = 48
        pos_dir = path_config.onet_pos_dir
        part_dir = path_config.onet_part_dir
        neg_dir = path_config.onet_neg_dir
    
    # 创建困难样本的生成路径
    for dir_path in [neg_dir, pos_dir, part_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    t_net(args.prefix,  # model param's file
          args.epoch,  # final epochs
          args.batch_size,  # test batch_size
          args.test_mode,  # test which model
          args.thresh,  # cls threshold
          args.min_face,  # min_face
          args.stride)
