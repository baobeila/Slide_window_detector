"""主要是对一个文件夹里的某些图片进行挑拣，进行数据增强"""
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import shutil
sdi_pre = os.listdir('./data/sdi_pre')
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
for index in range(1,24):
    #写循环
        #计算数字位数
        numlen = len(map(int, str(index)))
        #挑选某些数字，如24.jpg  把24挑选出来，对应出24.jpg进行数据增强
        train_sdi_pre = filter(lambda x:x[:numlen] ==str(index) , sdi_pre)
        for file_i in train_sdi_pre:

            sdi_pre_path = os.path.join('./data/spdi_pre',file_i)
            img = load_img(sdi_pre_path)  # this is a PIL image
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `preview/` directory
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir="./preview/", save_prefix='spdi_pre', save_format='jpeg'):
                i += 1
                if i > 20:#控制增强的数量
                    break  # otherwise the generator would loop indefinitely