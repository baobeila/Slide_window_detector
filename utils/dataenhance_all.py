"""单独对某个文件夹里所有图像增强，并对增强后的图片保存到文件夹"""
from keras_preprocessing.image import ImageDataGenerator
def generate_from_derictory(img_path, batch_size, class_mode=None):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.0,
        zoom_range=0.2,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        brightness_range=None,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.0
    )

    train_generator = train_datagen.flow_from_directory(
        directory=img_path,
        target_size=(64, 64),
        batch_size=batch_size,
        save_to_dir="./preview/",
        class_mode=class_mode)
    return train_generator

if __name__ == '__main__':

    i = 0
    data_gen = generate_from_derictory('./data/', 1, None)
    for im in data_gen:
        i += 1
        if i > 50:
            break  # 否则生成器会退出循环
