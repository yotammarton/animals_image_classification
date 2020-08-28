import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
import skimage
import tensorflow as tf
import pandas as pd
import os


def crop_and_resize(img, target_image_size=None):
    """
    crop an image and resize to target_image_size shape
    :param img: source RGB image ndarray (height,width,3)
    :param target_image_size: tuple (height,width)
    :return: cropped and sized image
    """
    if not target_image_size:
        target_image_size = (img.shape[0], img.shape[1])  # resize to the same original shape

    # TODO (YOTAM) possible change for the sizes (currently dropping 1/5 of the image from every side)
    cropped = tf.image.crop_to_bounding_box(img, offset_height=img.shape[0] // 5, offset_width=img.shape[1] // 5,
                                            target_height=3 * img.shape[0] // 5, target_width=3 * img.shape[1] // 5)

    cropped_resized = tf.image.resize(cropped, size=target_image_size,
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR).numpy()

    return cropped_resized


def augment_flip_lr(img):
    """
    flip image left right
    :param img: source RGB image ndarray (height,width,3)
    :return: flipped image
    """
    return np.fliplr(img)


def augment_flip_lr_crop(img):
    """
    flip image left right and crop to some part of it, than resize to the same shape of img
    :param img: source RGB image ndarray (height,width,3)
    :return: cropped part of the left-right flipped image, with the same dimensions of original image
    """
    # flip left - right and than  crop and resize image
    return crop_and_resize(augment_flip_lr(img))


def augment_add_noise(img):
    """
    add gaussian noise to the image
    :param img: source RGB image ndarray (height,width,3)
    :return: noised image with the same shape as img
    """
    # convert image values from [0,255] to floats and add noise
    noised_float = skimage.util.random_noise(img)

    # convert back to [0,255] with the noise
    noised = skimage.img_as_ubyte(noised_float)
    return noised


def voc_cats_dogs_images():
    """
    get the images of cats and dogs from PASCAL VOC
    filter images with both cat and dog

    returns df
    path      | cat/dog  | breed  | dataset
    ___________________________________
    file path | dog     | 3      | voc
    file path | cat     | 6      | voc
    :return: data frame with voc images paths
    """
    cat_path = os.path.join('VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'cat_trainval.txt')
    dog_path = os.path.join('VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'dog_trainval.txt')

    # a set for the images names for cats and dogs
    voc_cats_images_names = set()
    voc_dogs_images_names = set()

    for animal, path in {'cat': cat_path, 'dog': dog_path}.items():
        with open(path) as file:
            for line in file.readlines():
                splited = line.split()
                file_name, has_animal = splited[0] + '.jpg', splited[1]
                if has_animal == '1':
                    if animal == 'cat':
                        voc_cats_images_names.add(os.path.join('VOCdevkit', 'VOC2012', 'JPEGImages', file_name))
                    else:
                        voc_dogs_images_names.add(os.path.join('VOCdevkit', 'VOC2012', 'JPEGImages', file_name))

    # filter pictures containing both cats and dogs together
    intersection = voc_cats_images_names.intersection(voc_dogs_images_names)

    voc_cats_images_names -= intersection
    voc_dogs_images_names -= intersection

    df = pd.DataFrame(columns=['path', 'cat/dog', 'breed', 'dataset'])
    for cat in voc_cats_images_names:
        df = df.append({'path': cat, 'cat/dog': 'cat', 'breed': None, 'dataset': 'voc'}, ignore_index=True)

    for dog in voc_dogs_images_names:
        df = df.append({'path': dog, 'cat/dog': 'dog', 'breed': None, 'dataset': 'voc'}, ignore_index=True)

    return df[['path', 'cat/dog', 'breed', 'dataset']]


def oxford_cats_dogs_images():
    """
    path      | cat/dog  | breed  | dataset
    ___________________________________
    file path | dog     | 3      | oxford
    file path | cat     | 6      | oxford
    :return: data frame with oxford images paths
    """
    df = pd.DataFrame(columns=['path', 'cat/dog', 'breed', 'dataset'])
    path = os.path.join('annotations', 'list.txt')

    with open(path) as file:
        for line in file.readlines():
            if line[0] == '#':
                continue
            splited = line.split()
            file_name, breed, cat_or_dog = splited[0] + '.jpg', splited[1], splited[2]
            file_path = os.path.join('images', file_name)
            df = df.append({'path': file_path, 'cat/dog': 'cat' if cat_or_dog == '1' else 'dog',
                            'breed': breed, 'dataset': 'oxford'}, ignore_index=True)

    return df[['path', 'cat/dog', 'breed', 'dataset']]


def train_test_split_basic_model(df, advanced_model=False):
    if advanced_model:
        # get 70% of breed images for train and the rest for test
        # roughly 200 images per breed
        result_df = None
        for breed in set(df['breed']):
            # get the df for specific breed and shuffle it
            breed_df = df[df['breed'] == breed].sample(frac=1)
            test_size = int(len(breed_df) * 0.3)
            train_size = len(breed_df) - test_size
            breed_df['train/test'] = ['test'] * test_size + ['train'] * train_size
            if result_df is None:
                result_df = breed_df.copy()
            else:
                result_df = result_df.append(breed_df)
        return result_df.reset_index(drop=True)

    else:
        test_size = 1000
        cats = df[df['cat/dog'] == 'cat'].sample(frac=1, random_state=42)  # shuffle
        dogs = df[df['cat/dog'] == 'dog'].sample(frac=1, random_state=42)  # shuffle
        cats['train/test'] = ['test'] * test_size + ['train'] * (len(cats) - test_size)
        dogs['train/test'] = ['test'] * test_size + ['train'] * (len(dogs) - test_size)
        df = pd.concat([cats, dogs]).reset_index(drop=True)
        print(df.shape)
        return df


if __name__ == '__main__':
    """test augmentation"""
    # df = pd.read_csv('paths.csv')
    # smpl = df.sample(10)
    # for path in smpl['path']:
    #     imshow(imread(path))
    #     plt.show()
    #     imshow(augment_flip_lr_crop(imread(path)))
    #     plt.show()
    #     imshow(augment_add_noise(imread(path)))
    #     plt.show()

    """build the .csv for image paths for basic"""
    # df_voc = voc_cats_dogs_images()
    # df_oxford = oxford_cats_dogs_images()
    # df = pd.concat([df_voc, df_oxford])
    # df = train_test_split_basic_model(df)
    # df.to_csv('data_basic_model.csv')

    """build the .csv for image paths for advanced"""
    df_oxford = oxford_cats_dogs_images()
    df_oxford = train_test_split_basic_model(df_oxford, advanced_model=True)
    df_oxford.to_csv('data_advanced_model_linux.csv')

    pass
