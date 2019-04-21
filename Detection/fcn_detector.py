import tensorflow as tf


class FcnDetector(object):
    """ 利用全卷积检测网络对单张图片进行目标检测 """
    
    def __init__(self, net_factory, model_path):
        """ 全卷积检测网络初始化
        :param net_factory: 已定义的网络模型
        :param model_path: 模型参数文件路径
        """
        # 创建graph
        graph = tf.Graph()
        with graph.as_default():
            # 定义网络的操作和向量
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            # 得到网络模型：目标类别概率，bounding box，关键点预测值
            self.cls_prob, self.bbox_predict, _ = net_factory(image_reshape, training=False)
            
            # allow_soft_placement：如果你指定的设备不存在，允许TF自动分配设备
            # allow_growth：动态申请显存
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                         gpu_options=tf.GPUOptions(allow_growth=True)))
            # 检查checkpoint是否有效
            model_dict = '/'.join(model_path.split('/')[:-1])
            checkpoint = tf.train.get_checkpoint_state(model_dict)
            read_state = checkpoint and checkpoint.model_checkpoint_path
            assert read_state, "全卷积网络模型参数路径无效！"
            print("加载模型参数...")
            print(model_path)
            saver = tf.train.Saver()
            saver.restore(self.sess, model_path)
            print("模型参数完成加载!")
    
    def predict(self, data_batch):
        height, width, _ = data_batch.shape
        # print(height, width)
        cls_prob, bbox_predict = self.sess.run([self.cls_prob, self.bbox_predict],
                                               feed_dict={self.image_op: data_batch,
                                                          self.width_op: width, self.height_op: height})
        return cls_prob, bbox_predict
