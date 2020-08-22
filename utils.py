import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
import skimage
import tensorflow as tf
import pandas as pd


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


if __name__ == '__main__':
    # test augmentation
    df = pd.read_csv('paths.csv')
    smpl = df.sample(10)
    for path in smpl['path']:
        imshow(imread(path))
        plt.show()
        imshow(augment_flip_lr_crop(imread(path)))
        plt.show()
        imshow(augment_add_noise(imread(path)))
        plt.show()
