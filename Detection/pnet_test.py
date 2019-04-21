# -*- coding: utf-8 -*-
"""
Created on 2019/4/11
File pnet_test
@author:ZhengYuwei
功能：测试PNet网络：precision，recall，time
"""
import os
import argparse
import cv2
from Detection.fcn_detector import FcnDetector
from Detection.detector import Detector
from Detection.MtcnnDetector import MtcnnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
from prepare_data.utils import IoU
from train_models.MTCNN_config import config
from prepare_data.path_configure import PathConfiguration
import numpy as np
import pdb


class PNetTester(object):
    """ 测试PNet模型 """
    
    def __init__(self, models_path, image_path, ground_truth_file):
        """ 初始化：加载PNet模型、待测试图片路径及对应ground truth
        :param models_path: [PNet, RNet, ONet]模型路径
        :param image_path: 待测试图片路径
        :param ground_truth_file: 待测图片目标ground_truth文件
        """
        # 初始化PNet模型检测器
        pnet_model = None
        rnet_model = None
        onet_model = None
        if models_path[0] is not None:
            pnet_model = FcnDetector(P_Net, models_path[0])
        if models_path[1] is not None:
            rnet_model = Detector(R_Net, 24, 256, models_path[1])
        if models_path[2] is not None:
            onet_model = Detector(O_Net, 48, 16, models_path[2])
            
        self.detector = MtcnnDetector([pnet_model, rnet_model, onet_model], min_face_size=20, stride=2,
                                      threshold=[0.7, 0.7, 0.7], scale_factor=0.79, slide_window=False)
        # 初始化ground truth
        self.ground_map = dict()
        valid_image_path = list()
        with open(ground_truth_file, 'r') as truth_file:
            for ground_truth in truth_file:
                ground_truth = ground_truth.strip().split(' ')
                self.ground_map[ground_truth[0]] = np.array([float(_) for _ in ground_truth[1:]])
                valid_image_path.append(ground_truth[0])
                
        # 初始化图片加载器
        if os.path.isdir(image_path):
            images_path = PNetTester.search_file(image_path)
        elif os.path.isfile(image_path) and image_path.endswith('.jpg'):
            images_path = [image_path]
        
        self.images_path = list()
        for image_path in images_path:
            if os.path.basename(image_path) in valid_image_path:
                self.images_path.append(image_path)
        print('待检测图片数量：', len(self.images_path))
        self.test_loader = TestLoader(self.images_path)
        
        return
    
    @staticmethod
    def search_file(search_path):
        """在指定目录search_path下，递归目录搜索jpg文件
        :param search_path: 指定的搜索目录，如：./2018年收集的素材并已校正
        :return: 该目录下所有jpg文件的路径组成的list
        """
        jpg_path_list = list()
        # 获取：1.父目录绝对路径 2.所有文件夹名字（不含路径） 3.所有文件名字
        for root_path, dir_names, file_names in os.walk(search_path):
            # 收集符合条件的文件名
            for filename in file_names:
                if filename.endswith('.jpg') and filename.find(' ') == -1:
                    jpg_path_list.append(os.path.join(root_path, filename))
        return jpg_path_list

    def test(self, threshold, model_iter):
        """
        :param threshold:
        :param model_iter:
        :return:
        """
        all_boxes, landmark = self.detector.detect_images(self.test_loader)
        
        hard_samples = list()
        recall = 0  # 召回率：TP / (TP + TN)
        acc_pos = 0
        acc_all = 0
        precision = 0  # 精确率：TP / (TP + FP)
        save_path = os.path.join(os.path.dirname(self.images_path[0]), '..', 'result')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for index, image_path in enumerate(self.images_path):
            image_name = os.path.basename(image_path)
            ground_truth = self.ground_map[image_name]
            if len(all_boxes[index]) == 0:
                print('图片{}检测不到车牌'.format(image_name))
                continue
            # 计算iou，并画框
            iou = np.ones((len(all_boxes[index]),))
            gt_boxes = np.array([ground_truth])
            for j, box in enumerate(all_boxes[index]):
                iou[j] = IoU(box, gt_boxes)
            '''
            # 画图
            im = cv2.imread(image_path)
            for j, box in enumerate(all_boxes[index]):
                # if image_name == '20180929172716720_23609_dqp001_甘A5T470.jpg':
                #    pdb.set_trace()
                # 绘制iou大于阈值的pos框
                if iou[j] > threshold:
                    cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                  (255, 0, 0), 2)
                    cv2.putText(im, '{:s}|{:.2f}|{:.2f}'.format('p', box[4], iou[j]),
                                (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0))
                    for k in range(5):
                        cv2.circle(im, (landmark[index][j][2*k], landmark[index][j][2*k+1]), 1, (0, 0, 255), 4)
                    
            # 绘制ground truth
            cv2.rectangle(im, (int(ground_truth[0]), int(ground_truth[1])),
                          (int(ground_truth[2]), int(ground_truth[3])),
                          (0, 255, 0), 2)
            cv2.imwrite(os.path.join(save_path, os.path.splitext(image_name)[0] + '_' + model_iter + '.jpg'), im)
            
            print('IoU:\n', iou)
            print('average iou = {}'.format(sum(iou) / sum(iou != 0)))
            '''
            # 计算检测框iou大于阈值的平均精度
            if iou.max() > threshold:
                recall += 1
                acc_pos += np.mean(all_boxes[index][iou > threshold, 4])
                acc_all += np.mean(all_boxes[index][:, 4])
                precision += len(all_boxes[index][iou > threshold, 4]) / len(all_boxes[index][:, 4])
            else:
                hard_samples.append(image_path)

        precision /= recall
        acc_pos /= recall
        acc_all /= recall
        recall /= self.test_loader.size
        print('IoU threshold={}:'.format(threshold), 'precision={},'.format(precision),
              ' acc-pos={},'.format(acc_pos), 'acc-all={}'.format(acc_all), 'recall={}'.format(recall))
        return precision, acc_pos, acc_all, recall


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = config.VISIBLE_GPU
    path_config = PathConfiguration().config
    """ 对于PNet来说，重要的是召回率，再考虑准确度 """
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--iter', dest='iter', help='model iteration of PNet',
                        default=['14', '40', '20'], type=str)  # 14 40
    parser.add_argument('--image_path', dest='image_path', help='image path to test', type=str,
                        default=path_config.images_dir)
    parser.add_argument('--truth_file', dest='truth_file', help='image ground truth file', type=str,
                        default=path_config.point2_val_txt_path)
    parser.add_argument('--threshold', dest='threshold', help='threshold',
                        default=0.65, type=float)
    args = parser.parse_args()
    
    precisions = list()
    acc_alls = list()
    acc_poss = list()
    recalls = list()
    for i in ['10', '20', '30', '40', '50']:
        args.iter[2] = i
        pnet_model_path = list()
        pnet_model_path.append(path_config.pnet_landmark_model_path + '-' + args.iter[0])
        pnet_model_path.append(path_config.rnet_landmark_model_path + '-' + args.iter[1])
        # pnet_model_path.append(None)
        pnet_model_path.append(path_config.onet_landmark_model_path + '-' + args.iter[2])
        pnet_tester = PNetTester(pnet_model_path, args.image_path, args.truth_file)
        precision, accuracy_pos, accuracy_all, aver_recall = pnet_tester.test(args.threshold, args.iter[0])
        precisions.append(precision)
        acc_poss.append(accuracy_pos)
        acc_alls.append(accuracy_all)
        recalls.append(aver_recall)
    
    print('precision:{} \nacc-pos:{} \nacc-all:{} \nrecalls:{}'.format(precisions, acc_poss, acc_alls, recalls))
