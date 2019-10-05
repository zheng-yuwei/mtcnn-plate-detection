- gen_12net_data.py: 训练PNet的数据的采样代码
- gen_hard_example.py: 分别生成RNet和ONet的训练数据
- gen_imglist_xxnet.py: 分别将三个网络的三个任务（分类，回归，特征点检测）的数据汇总到一个文件中
- gen_xx_tfrecords.py: 分别生成3个网络的tfrecord,在这里需要注意：
  - PNet的训练数据(pos,neg,part,landmark)是混在一起的，生成了一个tfrecord
  - RNet和ONet的各自需要生成4个tfrecord(pos,neg,part,landmark)，因为要控制各部分的样本比例（1：3：1：1）
- loader.py: 迭代器，用于读取图片
- read_tfrecord_v2.py/tfrecord_utils.py: 用于读取tfrecord数据，并对其解析
- utils.py: 用于一些数据处理操作
- gen_landmark_tfrecords_aug_xx.py  
    用于生成特征点的数据，在这里并没有生成tfreord,只是对进行数据增强（随机镜像、随机旋转）
    此脚本的输入是trainImageList.txt,其中定义了文件的路径，人脸框的位置（x1,x2,y1,y2）,特征点的位置（x1,y1,,,,,x5,y5）
- BBox_utils.py/landmark_utils.py: 用于特征点处理