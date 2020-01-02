from model import dense_block, conv_block, convolutional_block, identity_block
import numpy as np
import keras.backend as K
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, \
    BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, \
    advanced_activations, AlphaDropout
from keras.models import Model, load_model
from keras import layers
from keras.initializers import glorot_uniform
from keras.preprocessing import image
from matplotlib import pyplot as plt

K.set_learning_phase(0)
from visualize import plot_bbox
import cv2
CLASSES = {'sdi_pre','spdi_pre','back_ground'}
name_dict = dict(zip(range(len(CLASSES)),CLASSES))
'''确定滑动窗口大小，进行滑窗
得到矩形框列表
crop
展平建立第二个模型
 p_24 = 0.8
得到输出，设置阈值概率，添加字典，保存框和概率'''


def read_big_model(X_input):  # 大模型中的卷积模块
    # Stage 1
    X = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv1",
               kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(X)

    # Stage 2
    X = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv2",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name="bn_conv2")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(X)

    # Stage 3
    ##Branch 1

    X_1 = convolutional_block(X, f=3, filters=[64, 64, 256], stage=3, block="a1", s=1)
    X_1 = identity_block(X_1, f=3, filters=[64, 64, 256], stage=3, block="a2")

    ##Branch 2
    nb_filter = 128  # 输入的 channel维度
    growth_rate = 16
    dropout_rate = 0.5
    weight_decay = 1e-5
    X_2, _ = dense_block(X, 8, nb_filter, growth_rate, bottleneck=True,
                         dropout_rate=dropout_rate, weight_decay=weight_decay)
    X = layers.add([X_1, X_2])

    # Stage 4
    X = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), name="conv4", padding="same",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name="bn_conv4")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(X)
    return X


def read_big_model_classify(inputs_sec, classes=2):
    # FC1
    X = Flatten()(inputs_sec)  # 展平
    X = Dense(4096, activation="relu", name="fc1" + str(4096), kernel_initializer=glorot_uniform(seed=0))(X)
    dropout_rate = 0.25
    X = Dropout(dropout_rate)(X)

    # FC2
    X = Dense(1024, activation="relu", name="fc2" + str(1024), kernel_initializer=glorot_uniform(seed=0))(X)
    dropout_rate = 0.25
    X = Dropout(dropout_rate)(X)

    predictions = Dense(classes, activation="softmax", name="softmax" + str(classes),
                        kernel_initializer=glorot_uniform(seed=0))(X)
    return predictions


# 单张图片读取，并预测
def read_model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(480, 640))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # 预测时的数据处理  归一化
    # amin, amax = x.min(), x.max()  # 求最大最小值
    # x = (x - amin) / (amax - amin)
    # rescale操作,保持与模型测试时相同的操作
    x *= 1. / 255
    feature_preds = model.predict(x)
    return feature_preds


# 进行滑动窗口预测
def slide_window(img, window_size, stride):
    # 对构建的金字塔图片，滑动窗口。
    # img：张量特征图，四维向量b h w c， window_size：滑动窗的大小，stride：步长。
    window_list = []
    w = img.shape[2]
    h = img.shape[1]
    if w <= window_size + stride or h <= window_size + stride:
        return None
    if len(img.shape) != 4:
        return None
    for i in range(int((w - window_size) / stride)):
        for j in range(int((h - window_size) / stride)):
            box = [j * stride, i * stride, j * stride + window_size, i * stride + window_size]

            window_list.append(box)
    return window_list


def nms(dets, prob_threshold):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 获得由大到小的分数索引
    score_index = np.argsort(scores)[::-1]

    keep = []

    while score_index.size > 0:
        max_index = score_index[0]
        # 最大的肯定是需要的框
        keep.append(max_index)
        xx1 = np.maximum(x1[max_index], x1[score_index[1:]])
        yy1 = np.maximum(y1[max_index], y1[score_index[1:]])
        xx2 = np.minimum(x2[max_index], x2[score_index[1:]])
        yy2 = np.minimum(y2[max_index], y2[score_index[1:]])

        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)

        union = width * height

        iou = union / (areas[max_index] + areas[score_index[1:]] - union)
        ids = np.where(iou <= prob_threshold)[0]
        # 因为算iou的时候没把第一个参考框索引考虑进来，所以这里都要+1
        score_index = score_index[ids + 1]
    return keep


if __name__ == '__main__':
    # 将输入定义为维度大小为 input_shape的张量
    input_shape = (480, 640, 3)
    window_size = 8
    stride = 1
    X_input = Input(input_shape)  # 构造输入
    X = read_big_model(X_input)  # 定义模型输入输出
    # 建立第一个modle

    model_one = Model(inputs=X_input, outputs=X)
    model_one.load_weights('Weight.h5', by_name=True)  # 读取训练好模型的网络参数
    # 读入一张图片
    img_path = './test/SDI_2.bmp'
    # 模型预测，得到特征图（1，60，80，128）8倍压缩
    feature_map = read_model_predict(img_path, model_one)
    # 窗口7*7  步长1
    slidereturn = slide_window(feature_map, window_size, stride)
    # 创建字典保存
    newdict = {}
    # 定义输入大小，由64*64，在该阶段的大小决定
    fea_input = Input((8, 8, 128))
    # 创建第二个模型，定义输入输出
    predsX = read_big_model_classify(fea_input, classes=3)
    # 模型建立
    model_two = Model(inputs=fea_input, outputs=predsX)

    # 名字相同的节点，进行权重加载
    model_two.load_weights('Weight.h5', by_name=True)
    # weight_Dense_1,bias_Dense_1 = model_one.get_layer('conv1').get_weights()
    # print(weight_Dense_1)

    ima = cv2.imread(img_path)
    ima_copy = ima.copy()
    i = 0
    # 定义字典，字典的key为缺陷类型，字典的value为对应的候选框,进行统计
    proposal = {}
    for index, bndbox in enumerate(slidereturn):
        # feature_map[:,0:60,0:80,:]
        # （N  C  H   W）注意格式
        region = feature_map[:, bndbox[0]:bndbox[2], bndbox[1]:bndbox[3], :]
        # region = tf.image.crop_to_bounding_box(feature_map,bndbox[1],bn   dbox[0],bndbox[3]-bndbox[1] ,bndbox[2]-bndbox[0])
        predsX = model_two.predict(region, verbose=0)
        # 对预测值解码，翻译为缺陷
        biao = np.argmax(predsX, axis=1)[0]  # [2]
        newdict[index] = biao  # 按列方向搜索最大值,找到标签对应的编号
        # {0:'sdi_pre',1:'spdi_pre',2:'back_ground'}
        if biao == 0 and predsX[0][biao] > 0.9:  # 找到概率值
            bbox = [int(bndbox[1]) * 8, int(bndbox[0]) * 8, int(bndbox[3]) * 8, int(bndbox[2]) * 8, predsX[0][biao]]
            proposal.setdefault(name_dict[biao], []).append(bbox)
        elif biao == 1 and predsX[0][biao] > 0.99:
            bbox = [int(bndbox[1]) * 8, int(bndbox[0]) * 8, int(bndbox[3]) * 8, int(bndbox[2]) * 8, predsX[0][biao]]
            proposal.setdefault(name_dict[biao], []).append(bbox)
    # 对每一个类别分别进行NMS；一次读取一对键值（即某个类别的所有框）
    for object_name, bbox in proposal.items():
        dets = np.array(bbox, dtype=np.float)
        # 可视化代码开始，对比nms前后
        plt.figure(1)
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        plt.sca(ax1)
        plot_bbox(dets, 'k')# before nms
        ######
        keep = nms(dets, 0.1)
        plt.sca(ax2)
        plot_bbox(dets[keep], 'r')  # after nms
        plt.show()  # 加该句图片才进行显示
        #######
        c_dets = dets[keep, :]
        proposal[object_name] = c_dets

        # Draw parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

    for class_id, bboxes in proposal.items():
        # (x1, y1), (x2, y2),设置左上顶点和右下顶点.astype("int")
        for id_num, sub_box in enumerate(bboxes):
            (w, h), baseline = cv2.getTextSize("{:.2f}".format(sub_box[4]), font, font_scale, thickness)
            cv2.rectangle(ima_copy, (int(sub_box[0]), int(sub_box[1]) - (2 * baseline + 5)),
                          (int(sub_box[0]) + w, int(sub_box[1])), (0, 255, 255), -1)
            cv2.rectangle(ima_copy, (int(sub_box[0]), int(sub_box[1])), (int(sub_box[2]), int(sub_box[3])), (0, 255, 0),
                          2)
            cv2.putText(ima_copy, "{:.2f}".format(sub_box[4]), (int(sub_box[0]), int(sub_box[1])), font, font_scale,
                        (0, 0, 0), thickness)
    cv2.imwrite('./output/output_test/SDI_2.jpg', ima_copy)
