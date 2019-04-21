import cv2
import time
import numpy as np
from Detection.nms import py_nms


class MtcnnDetector(object):

    def __init__(self, detectors, min_face_size=20, stride=2, threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.79, slide_window=False):
        """ MTCNN模型参数初始化
        :param detectors: 检测模型list
        :param min_face_size: 目标最小尺度
        :param stride: 全卷积网络PNet的stride
        :param threshold: 不同检测模型检测输出（作为下一个模型输入）的class阈值
        :param scale_factor: 全卷积网络PNet的图像金字塔缩放比例
        :param slide_window: 是否滑动窗口
        """
        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor
        self.slide_window = slide_window

    def convert_to_square(self, bbox):
        """ 将bounding boxes转换为矩形（PNet或RNet输出给下一个网络时需要）
        :param bbox: 输入的bounding box， num * 5
        :return 矩形bounding boxes
        """
        square_bbox = bbox.copy()
        # 将短边按长边扩展区域
        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h, w)
        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox

    def calibrate_box(self, bbox, reg):
        """
            calibrate bboxes
        Parameters:
        ----------
            bbox: numpy array, shape n x 5
                input bboxes
            reg:  numpy array, shape n x 4
                bboxes adjustment
        Returns:
        -------
            bboxes after refinement
        """

        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c

    def generate_bbox(self, cls_map, reg, scale, threshold):
        """
            generate bbox from feature cls_map according to the threshold
        Parameters:
        ----------
            cls_map: numpy array , n x m 
                detect score for each position
            reg: numpy array , n x m x 4
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        stride = 2
        # stride = 4
        cellsize = 12
        # cellsize = 25

        # index of class_prob larger than threshold
        t_index = np.where(cls_map > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        # offset
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg])

        return boundingbox.T

    @staticmethod
    def processed_image(img, scale):
        """ rescale/resize the image according to the scale
        :param img: image
        :param scale:
        :return: resized image
        """
        height, width, _ = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)  # resized image
        # 因为在训练时，做了归一化，可参考训练时的tfrecord数据加载，read_tfrecord_v2.py 33行
        img_resized = (img_resized - 127.5) / 128
        return img_resized

    @staticmethod
    def pad(bboxes, w, h):
        """
            pad the the bboxes, also restrict the size of it
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def detect_pnet(self, im):
        """ 通过PNet获取单张图片的目标检测候选框
        :param im: 图像数据
        :return boxes: 通过长宽归一化后的候选框
        :return boxes_c: 根据长宽恢复到原始图像尺度的候选框
        """
        net_size = 12
        # 初始化scale：将输入图像做尺度变换，以输入PNet进行检测
        # 初始scale能检测最小尺度目标，后续scale逐渐变小，图片resize为更小尺度，PNet相对而言感受域变大，检测更大目标
        current_scale = float(net_size) / self.min_face_size
        # 根据初始尺度做图像缩小，以放入PNet做检测
        im_resized = self.processed_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape
        # 全卷积网络
        all_boxes = list()
        while min(current_height, current_width) > net_size:
            # PNet预测：类别(H * W * 2)，bounding boxes(H * W * 4)
            cls_map, detected_boxes = self.pnet_detector.predict(im_resized)
            # bounding boxes恢复到原图像尺寸的坐标及off_set
            # 预测的bounding boxes数量 * 9：x1, y1, x2, y2, score, x1_offset, y1_offset, x2_offset, y2_offset
            boxes = self.generate_bbox(cls_map[:, :, 1], detected_boxes, current_scale, self.thresh[0])
            # 根据预设的尺度做图像缩小，以放入PNet做检测
            current_scale *= self.scale_factor
            im_resized = self.processed_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue
            # 做非极大值抑制，筛选候选框
            keep = py_nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)
        if len(all_boxes) == 0:
            return None, None, None
        
        # 先nms，再做offset矫正回归框
        # 对PNet所有尺度检测得到的候选框做非极大值抑制，筛选候选框
        all_boxes = np.vstack(all_boxes)
        keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]
        # 根据offset计算实际预测的候选框坐标
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T
        '''
        # 先做offset矫正回归框，再做nms
        all_boxes = np.vstack(all_boxes)
        # 根据offset计算实际预测的候选框坐标
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T
        # 对PNet所有尺度检测得到的候选框做非极大值抑制，筛选候选框
        keep = py_nms(boxes_c[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]
        boxes_c = boxes_c[keep]
        '''
        return boxes, boxes_c, None

    def detect_rnet(self, im, dets):
        """ 获取PNet候选框区域图像进行RNet检测
        :param im: 原始图像
        :param dets: PNet检测的候选框
        :return boxes: 通过长宽归一化后的候选框
        :return boxes_c: 根据长宽恢复到原始图像尺度的候选框
        """
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (24, 24)) - 127.5) / 128
        # cls_scores : num_data*2
        # reg: num_data*4
        # landmark: num_data*10
        cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            # landmark = landmark[keep_inds]
        else:
            return None, None, None
        
        # 先nms，再做offset矫正回归框
        keep = py_nms(boxes, 0.6)
        boxes = boxes[keep]
        boxes_c = self.calibrate_box(boxes, reg[keep])
        '''
        # 先做offset矫正回归框，再做nms（RNet这个召回率高点）
        boxes_c = self.calibrate_box(boxes, reg)
        keep = py_nms(boxes_c, 0.6)
        boxes = boxes[keep]
        boxes_c = boxes_c[keep]
        '''
        return boxes, boxes_c, None

    def detect_onet(self, im, dets):
        """Get face candidates using onet

        Parameters:
        ----------
        im: numpy array
            input image array
        dets: numpy array
            detection results of rnet

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        h, w, c = im.shape
        dets = self.convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128

        cls_scores, reg, landmark = self.onet_detector.predict(cropped_ims)
        # prob belongs to face
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[2])[0]
        if len(keep_inds) > 0:
            # pickout filtered box
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None, None

        # width
        w = boxes[:, 2] - boxes[:, 0] + 1
        # height
        h = boxes[:, 3] - boxes[:, 1] + 1
        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        boxes_c = self.calibrate_box(boxes, reg)

        # 先nms，再做offset矫正回归框，避免同一位置/目标输出多个框
        boxes = boxes[py_nms(boxes, 0.6, "Minimum")]
        keep = py_nms(boxes_c, 0.6, "Minimum")
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes, boxes_c, landmark

    # use for video
    def detect(self, img):
        """Detect face over image
        """
        # PNet
        if self.pnet_detector:
            _, boxes_c, landmark = self.detect_pnet(img)
            if boxes_c is None:
                return np.array([]), np.array([])
        # RNet
        if self.rnet_detector:
            _, boxes_c, landmark = self.detect_rnet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])
        # ONet
        if self.onet_detector:
            _, boxes_c, landmark = self.detect_onet(img, boxes_c)
            if boxes_c is None:
                return np.array([]), np.array([])

        return boxes_c, landmark

    def detect_images(self, test_data):
        """ 循环检测多张图片 """
        bounding_boxes = []
        landmarks = []
        batch_idx = 0
        s_time = time.time()
        for single_image in test_data:
            batch_idx += 1
            if batch_idx % 100 == 0:
                c_time = (time.time() - s_time)/100
                print("检测进度：{}/{}".format(batch_idx, test_data.size))
                print('每张图耗时：{} s'.format(c_time))
                s_time = time.time()
            # result_boxes * 9, result_boxes * 10
            boxes_c, landmark = self.detect(single_image)
            bounding_boxes.append(boxes_c)
            landmarks.append(landmark)
        print('预测的bounding boxes数量:', len(bounding_boxes))
        # 图片数 * [result_boxes * 9], 图片数 * [result_boxes * 10]
        return bounding_boxes, landmarks

    def detect_single_image(self, im):
        all_boxes = []
        landmarks = []
        if self.pnet_detector:
            # ignore landmark
            boxes, boxes_c, landmark = self.detect_pnet(im)
            if boxes_c is None:
                print("boxes_c is None...")
                all_boxes.append(np.array([]))
                # pay attention
                landmarks.append(np.array([]))
        # rnet
        if self.rnet_detector and boxes_c is not None:
            # ignore landmark
            boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)
            if boxes_c is None:
                all_boxes.append(np.array([]))
                landmarks.append(np.array([]))
        # onet
        if self.onet_detector and boxes_c is not None:
            boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)
            if boxes_c is None:
                all_boxes.append(np.array([]))
                landmarks.append(np.array([]))
        all_boxes.append(boxes_c)
        landmarks.append(landmark)
        return all_boxes, landmarks
