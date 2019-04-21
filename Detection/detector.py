import tensorflow as tf
import numpy as np


class Detector(object):
    """ RNet或ONet检测模型 """
    
    def __init__(self, net_factory, data_size, batch_size, model_path):
        """ 初始化检测模型graph
        :param net_factory:
        :param data_size: 24 or 48
        :param batch_size:
        :param model_path:
        """
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[batch_size, data_size, data_size, 3], name='input_image')
            # figure out landmark
            self.cls_prob, self.bbox_pred, self.landmark_pred = net_factory(self.image_op, training=False)
            self.sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            
            # 检查checkpoint是否有效
            model_dict = '/'.join(model_path.split('/')[:-1])
            checkpoint = tf.train.get_checkpoint_state(model_dict)
            read_state = checkpoint and checkpoint.model_checkpoint_path
            assert read_state, "the params dictionary is not valid"
            print("加载模型参数...")
            print(model_path)
            saver = tf.train.Saver()
            saver.restore(self.sess, model_path)
            print("模型参数完成加载!")
            
        self.data_size = data_size
        self.batch_size = batch_size
    
    def predict(self, data_batch):
        """ RNet 或 ONet模型 批量检测
        :param data_batch: 批量图像数据
        :return:
        """
        # data_batch: N x 3 x data_size x data_size
        batch_size = self.batch_size
        
        # 将待检测的批量图片数据，分割为合适的小批量图片数据，已进行后续检测
        n = data_batch.shape[0]
        mini_batch = []
        cur = 0
        while cur < n:
            mini_batch.append(data_batch[cur:min(cur + batch_size, n), :, :, :])
            cur += batch_size
        # 每一个批量的预测结果
        cls_prob_list = []
        bbox_predict_list = []
        landmark_predict_list = []
        for idx, data in enumerate(mini_batch):
            m = data.shape[0]
            real_size = self.batch_size
            # 最后一批如果不满小批量，填补为一整个批量（因为已定义graph输入图像batch的限制）
            if m < batch_size:
                keep_inds = np.arange(m)
                # gap (difference)
                gap = self.batch_size - m
                while gap >= len(keep_inds):
                    gap -= len(keep_inds)
                    keep_inds = np.concatenate((keep_inds, keep_inds))
                if gap != 0:
                    keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
                data = data[keep_inds]
                real_size = m
            # 检测结果
            cls_prob, bbox_pred, landmark_pred = self.sess.run([self.cls_prob, self.bbox_pred, self.landmark_pred],
                                                               feed_dict={self.image_op: data})
            
            cls_prob_list.append(cls_prob[:real_size])  # num_batch * batch_size * 2
            bbox_predict_list.append(bbox_pred[:real_size])  # num_batch * batch_size * 4
            landmark_predict_list.append(landmark_pred[:real_size])  # num_batch * batch_size * 10
        # 调整输出格式num_of_data * 2, num_of_data * 4, num_of_data * 10
        cls_prob_list = np.concatenate(cls_prob_list, axis=0)
        bbox_predict_list = np.concatenate(bbox_predict_list, axis=0)
        landmark_predict_list = np.concatenate(landmark_predict_list, axis=0)
        return cls_prob_list, bbox_predict_list, landmark_predict_list
