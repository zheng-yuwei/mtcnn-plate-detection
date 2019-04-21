import numpy as np


def IoU(box, boxes):
    """ 计算预测bounding box和ground truth的bounding boxes间的IoUs
    :param box: 一个预测bounding box，(5，):x1, y1, x2, y2, score
    :param boxes: 多个ground truth的bounding boxes， (n, 4): x1, y1, x2, y2
    :return ovr: IoUs, shape (n, )
    """
    # 计算各种区域面积
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    # 计算相交部分面积
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    
    ovr = inter / (box_area + area - inter)
    return ovr


def square_IoU(box, boxes):
    """ 计算预测bounding box和ground truth的bounding boxes间的IoUs
    :param box: 一个预测bounding box，(5，):x1, y1, x2, y2, score
    :param boxes: 多个ground truth的bounding boxes， (n, 4): x1, y1, x2, y2
    :return ovr: IoUs, shape (n, )
    """
    square_boxes = convert_to_square(boxes)
    # 计算各种区域面积
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (square_boxes[:, 2] - square_boxes[:, 0] + 1) * (square_boxes[:, 3] - square_boxes[:, 1] + 1)
    # 计算相交部分面积
    xx1 = np.maximum(box[0], square_boxes[:, 0])
    yy1 = np.maximum(box[1], square_boxes[:, 1])
    xx2 = np.minimum(box[2], square_boxes[:, 2])
    yy2 = np.minimum(box[3], square_boxes[:, 3])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    
    ovr = inter / (box_area + area - inter)
    return ovr
    

def convert_to_square(bbox):
    """ 将bounding boxes转为方形
    :param bbox: numpy array , shape n x 5
    :return square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox
