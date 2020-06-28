from keras.preprocessing.image import (
    ImageDataGenerator, array_to_img, img_to_array, load_img)
import os
import shutil
import glob
import os.path


def augment_data(start_path, images, save_path, growth_ratio, label):
    datagen = ImageDataGenerator(
        rotation_range=10,
        shear_range=0.02,
        zoom_range=0.01,
        horizontal_flip=True,
        brightness_range=(0.2, 0.4))

    for file_index, imagepath in enumerate(images):
        print("Status:", file_index+1, "/", len(images), end="\r")
        img = load_img(imagepath)
        array_img = img_to_array(img)
        reshaped_img = array_img.reshape((1, ) + array_img.shape)
        for batch_index, batch in enumerate(datagen.flow(reshaped_img, batch_size=1, save_to_dir=save_path, save_prefix=imagepath.strip(start_path.strip(label)).strip('.jpg'), save_format='jpg')):
            if batch_index >= growth_ratio-2:
                break
        shutil.copy(imagepath, save_path+"/"+label)

    print("Status:", file_index+1, "/", len(images))
    print(" saved:", len(images)*growth_ratio, "images",
          growth_ratio, "times more images than original base")
