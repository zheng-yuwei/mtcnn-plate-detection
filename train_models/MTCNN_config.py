# coding:utf-8

from easydict import EasyDict

config = EasyDict()

config.BATCH_SIZE = 384
config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7
config.GPU = '2,3'
config.VISIBLE_GPU = '2,3'
config.EPS = 1e-14
config.LR_EPOCH = [6, 14, 20]  # [6, 14, 20]
