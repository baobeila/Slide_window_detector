#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 构建基于ResNet与DenseNet的神经网络实现缺陷图像的检测分类
# 深度学习框架 Keras中实现,参考CSDN博客，深度学习实现工业零件检测。
# TODO ：在使用生成器情况下如何显示梯度直方图，利用全卷积网络来做，增大共享区域
'''
生成器输入数据进行训练，Focal  loss 处理数据不均衡问题，（padding valid 效果更好点）
'''
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, \
    BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, \
    advanced_activations, AlphaDropout
from keras.models import Model
from keras.initializers import glorot_uniform
from keras.regularizers import l2
from keras.layers.merge import concatenate
import math
from sklearn.model_selection import train_test_split
import scipy.misc
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import keras
from keras.optimizers import SGD, Adam
# 数据模块
from utils.dataset import load_data
import matplotlib.pyplot as plt

K.set_image_data_format("channels_last")
# training mode  0 = test, 1 = train
K.set_learning_phase(1)
batch_size = 64

nb_train_samples = 940
epochs = 20


# 重新定义生成器中的预处理方法
class FixedImageDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        if self.featurewise_center:
            x = ((x / 255.) - 0.5) * 2.
        return x


# 恒等模块——identity_block
def identity_block(X, f, filters, stage, block):
    """
    三层的恒等残差块
    param :
    X -- 输入的张量，维度为（m, n_H_prev, n_W_prev, n_C_prev）
    f -- 整数，指定主路径的中间 CONV 窗口的形状（过滤器大小）
    filters -- python整数列表，定义主路径的CONV层中的过滤器
    stage -- 整数，用于命名层，取决于它们在网络中的位置
    block --字符串/字符，用于命名层，取决于它们在网络中的位置
    return:
    X -- 三层的恒等残差块的输出，维度为：(n_H, n_W, n_C)
    """
    # 定义基本的名字
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # 过滤器
    F1, F2, F3 = filters

    # 保存输入值,后面将需要添加回主路径
    X_shortcut = X

    # 主路径第一部分
    # 卷积核大小除以2，向下取整
    # X = ZeroPadding2D(padding=(1, 1), dim_ordering='tf')(X)
    # model.add(ZeroPadding2D((1, 1), batch_input_shape=(1, 4, 4, 1)))#序列模型加padding
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="same",
               name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    # 主路径第二部分
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    # 主路径第三部分
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="same",
               name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    # 主路径最后部分,为主路径添加shortcut并通过relu激活
    X = layers.add([X, X_shortcut])
    # X = Activation("relu")(X)
    X = advanced_activations.LeakyReLU(alpha=0.3)(X)
    return X


# 卷积残差块——convolutional_block
def convolutional_block(X, f, filters, stage, block, s=2):
    """
    param :
    X -- 输入的张量，维度为（m, n_H_prev, n_W_prev, n_C_prev）
    f -- 整数，指定主路径的中间 CONV 窗口的形状（过滤器大小，ResNet中f=3）
    filters -- python整数列表，定义主路径的CONV层中过滤器的数目
    stage -- 整数，用于命名层，取决于它们在网络中的位置
    block --字符串/字符，用于命名层，取决于它们在网络中的位置
    s -- 整数，指定使用的步幅
    return:
    X -- 卷积残差块的输出，维度为：(n_H, n_W, n_C)
    """
    # 定义基本的名字
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # 过滤器
    F1, F2, F3 = filters

    # 保存输入值,后面将需要添加回主路径
    X_shortcut = X

    # 主路径第一部分
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding="same",
               name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)

    # 主路径第二部分
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)

    # 主路径第三部分
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="same",
               name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    # shortcut路径
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding="same",
                        name=conv_name_base + "1", kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + "1")(X_shortcut)

    # 主路径最后部分,为主路径添加shortcut并通过relu激活
    X = layers.add([X, X_shortcut])
    # X = Activation("relu")(X)
    X = advanced_activations.LeakyReLU(alpha=0.3)(X)
    return X


# densenet 中相关的模块
def conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
        Args:
            ip: Input keras tensor
            nb_filter: number of filters
            bottleneck: add bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
        Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''
    concat_axis = 1 if K.image_data_format() == 'channel_first' else -1  # 特征轴

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    # 是否使用瓶颈层，也就是使用1x1的卷继层将特征图的通道数进行压缩，先增加到64通道后进行压缩到16
    if bottleneck:
        inter_channel = nb_filter * 4  # 16*4
        x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                grow_nb_filters=True, return_concat_list=False):
    '''Build a dense_block where the output of ench conv_block is fed t subsequent ones
        Args:
            x: keras tensor
            nb_layser: the number of layers of conv_block to append to the model
            nb_filter: number of filters
            growth_rate: growth rate#每次卷积时使用的卷积核个数
            bottleneck: bottleneck block
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
            return_concat_list: return the list of feature maps along with the actual output
        Returns:
            keras tensor with nb_layers of conv_block appened   x=1 growth_rate=2  x=1+2 growth_rate=3 x=1+2+3
    '''

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x_list = [x]

    for i in range(nb_layers):
        cb = conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        x_list.append(cb)
        x = concatenate([x, cb], axis=concat_axis)  # x在每次循环中始终维护一个全局状态

        if grow_nb_filters:
            nb_filter += growth_rate

    if return_concat_list:
        return x, nb_filter, x_list
    else:
        return x, nb_filter


def build_modle(input_shape=(64, 64, 3), classes=2):
    """
    :param input_shape:裁剪后的数据集图片维度
    :param classes: 整数，分类的数目
    :return:Keras中的模型实例
    Conv1：64x3x3
    Conv2：128x3x3
    """
    # 将输入定义为维度大小为 input_shape的张量
    X_input = Input(input_shape)
    # Stage 1
    X = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), name="conv1", kernel_initializer=glorot_uniform(seed=0),
               padding='same')(X_input)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(X)

    # Stage 2
    X = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), name="conv2", kernel_initializer=glorot_uniform(seed=0),
               padding='same')(X)
    X = BatchNormalization(axis=3, name="bn_conv2")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(X)

    # Stage 3
    ##Branch 1

    X_1 = convolutional_block(X, f=3, filters=[64, 64, 256], stage=3, block="a1", s=1)
    X_1 = identity_block(X_1, f=3, filters=[64, 64, 256], stage=3, block="a2")

    ##Branch 2,参数设置
    nb_filter = 128  # 输入的 channel维度
    growth_rate = 16
    dropout_rate = 0.5
    weight_decay = 1e-5
    X_2, _ = dense_block(X, 8, nb_filter, growth_rate, bottleneck=True,
                         dropout_rate=dropout_rate, weight_decay=weight_decay)
    X = layers.add([X_1, X_2])

    # Stage 4
    X = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", name="conv4",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name="bn_conv4")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(X)

    # FC1
    X = Flatten()(X)  # 展平
    # X = Dense(4096, activation="relu", name="fc1" + str(4096), kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(4096, activation="selu", name="fc1" + str(4096), kernel_initializer='lecun_normal',
              bias_initializer='lecun_normal')(X)
    dropout_rate = 0.25
    X = AlphaDropout(dropout_rate)(X)

    # FC2
    X = Dense(1024, activation="relu", name="fc2" + str(1024), kernel_initializer='lecun_normal',
              bias_initializer='lecun_normal')(X)
    dropout_rate = 0.25
    X = AlphaDropout(dropout_rate)(X)
    # 二分类为1
    X = Dense(classes, activation="softmax", name="softmax" + str(classes), kernel_initializer=glorot_uniform(seed=0))(
        X)

    # 创建模型
    model = Model(inputs=X_input, outputs=X, name="My_modle")
    return model


def focal_loss_fixed(y_true, y_pred, alpha=0.25, gamma=2):  # 多分类
    # tensorflow backend, alpha and gamma are hyper-parameters which can set by you
    eps = 1e-12
    y_pred = K.clip(y_pred, eps, 1. - eps)  # improve the stability of the focal loss
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    # return -K.sum(alpha * K.pow(tf.clip_by_value(1. - pt_1, 1e-8, 1.0), gamma) * K.log(
    #     tf.clip_by_value(pt_1, 1e-8, 1.0))) - K.sum(
    #     (1 - alpha) * K.pow(tf.clip_by_value(pt_0, 1e-8, 1.0), gamma) * K.log(tf.clip_by_value(1. - pt_0, 1e-8, 1.0)))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


# def focal_loss(y_true, y_pred,alpha=0.25,gamma=2):#单分类focal  loss
#     pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#     return -K.sum(alpha*K.pow(tf.clip_by_value(1. - pt_1, 1e-8, 1.0), gamma) * K.log(tf.clip_by_value(pt_1, 1e-8, 1.0)))

# LossHistory类，保存loss（err）和acc（score）
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        # 创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')  # plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)  # 设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')  # 给x，y轴加注释
        plt.legend(loc="upper right")  # 设置图例显示位置
        # plt.show()
        plt.savefig('./output/acc-loss.jpg')


if __name__ == '__main__':
    train_path = './data/val/'
    images, labels = load_data(train_path)
    # 主要是为了获得验证数据，验证数据以元组形式输入
    # 之所以不用生成器作为验证数据，是为了能够显示出直方图信息，以及梯度的相关信息
    #设置为0.95增大验证数据集数量
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.95, random_state=0)

    # 编译模型来配置学习过程
    train_img_path = './data/train/'
    # val_img_path = './data/train/'
    history = LossHistory()
    sgd = SGD(lr=0.0001, momentum=0.98, decay=0.99, nesterov=False)
    # adam = Adam(lr=0.0003)
    # 运行构建的模型图
    model = build_modle(input_shape=(64, 64, 3), classes=3)
    model.compile(
        # optimizer=adam,
        optimizer=sgd,
        # loss=focal_loss,#二分类时使用
        loss=focal_loss_fixed,
        # loss='categorical_crossentropy',  # 多分类交叉熵
        metrics=["accuracy"])
    # 显示模型配置信息
    model.summary()
    train_datagen = FixedImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        featurewise_center=True,
        samplewise_center=True,
        featurewise_std_normalization=True,
        samplewise_std_normalization=True,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        brightness_range=None,
        channel_shift_range=0.2,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.0
    )
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        directory=train_img_path,
        target_size=(64, 64),
        batch_size=batch_size,
        # save_to_dir= "./output/preview/",#增强后的图片保存起来，用以可视化
        class_mode='categorical',  # 二分类问题该值可设为‘binary’
        classes=['sdi_pre', 'spdi_pre', 'to_crop']  # 注意文件夹次序，字典序排序
    )
    # validation_generator = test_datagen.flow_from_directory(val_img_path,
    #                                                         target_size=(64, 64),
    #                                                         batch_size=32,
    #                                                         class_mode='categorical')
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./output/Graph',
                                             histogram_freq=1,  # 直方图标志
                                             # 设置为1，进行直方图计算
                                             write_graph=True,
                                             write_grads=True,  # 可视化梯度直方图
                                             write_images=True)
    # 训练模型
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[tbCallBack, history],
        # 当打印直方图时，验证数据不能是一个生成器，但可以是一个形如（inputs,targets）的tuple
        # validation_data=validation_generator,
        validation_data=(x_test, y_test),
        # validation_steps=nb_validation_samples // batch_size
        # 当validation_data为生成器时，本参数指定验证集的生成器返回次数
        # validation_steps=nb_train_samples // batch_size
        # validation_steps = math.ceil(nb_train_samples / batch_size)
    )
    # Keras自动定义类别的编号,{sdi_pre:0 spdi_pre:1,to_crop:2},注意对应
    check_label = train_generator.class_indices
    print(check_label)
    # model_name = 'Model_Weight.h5'
    # 完整地保存整个模型，将Keras模型和权重保存在一个HDF5文件中
    # model.save(model_name)
    model.save_weights('./output/Weight.h5')  # 保存模型的权重
    # 绘制损失，准确率曲线
    history.loss_plot('epoch')
