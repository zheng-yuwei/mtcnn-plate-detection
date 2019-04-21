import numpy as np
import cv2


class TestLoader(object):
    """ 图片loader对象 """
    
    def __init__(self, imdb, batch_size=1, shuffle=False):
        """ 初始化
        :param imdb: 待加载图片路径列表
        :param batch_size: 每次加载图片的批量
        :param shuffle: 是否对路径列表进行shuffle
        """
        self.imdb = imdb
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(imdb)  # num of data
        
        self.cur = 0
        self.data = None
        self.label = None

        self.reset()
        self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.imdb)

    def __next__(self):
        return self.next()
    
    def iter_next(self):
        return self.cur + self.batch_size <= self.size
    
    def __iter__(self):
        return self

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        imdb = self.imdb[self.cur]
        im = cv2.imread(imdb)
        self.data = im
