import argparse
import string
import os
import cv2
import imutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import utils


def image_to_feature_vector(image, size=(100, 100)):
    # resize the image to a fixed size, then flatten the image into a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(32, 32, 32)):
    # extract a 3D color histogram from the HSV color space using the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    # otherwise, perform "in place" normalization in OpenCV 3
    else:
        cv2.normalize(hist, hist)

    return hist.flatten()


def extract_SIFT(image, num_of_features=30):
    image = cv2.resize(image, (150, 150))  ## TODO change?
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_of_features)
    key_points, descriptors = sift.detectAndCompute(gray, None)  # None is for no-mask

    if len(key_points) > num_of_features:
        key_points = key_points[:num_of_features]
        descriptors = descriptors[:num_of_features]
        # sift_image = cv2.drawKeypoints(gray, key_points, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # plt.imshow(sift_image)
        # plt.show()
    elif len(key_points) < num_of_features:
        # key_points = [0]*num_of_features  # TODO this is not the right type. but it will not be used
        descriptors = np.zeros((num_of_features, 128))
        # print(f'attention, only {len(key_points)} SIFT-key-points')
    return descriptors.flatten()


def Classify(features, set_type, labels, c_of_svm):
    # print(set_type)
    train_features = [feature for index, feature in enumerate(features) if set_type[index] == 'train']
    test_features = [feature for index, feature in enumerate(features) if set_type[index] == 'test']
    train_labels = [label for index, label in enumerate(labels) if set_type[index] == 'train']
    test_labels = [label for index, label in enumerate(labels) if set_type[index] == 'test']

    print(f'number of train featuers: {len(train_features)} and train labels: {len(train_labels)}')
    print(f'number of test featuers: {len(test_features)} and test labels: {len(test_labels)}')

    acc, k = KNN(3, train_features, train_labels, test_features, test_labels)
    print(f'    {k}NN - accuracy: {(round(acc*100, 4))}')

    acc = SVM(train_features, train_labels, test_features, test_labels, c_of_svm)
    print(f'    SVM - accuracy: {(round(acc*100, 4))}')

    # acc = neural_network_MLP(train_features, train_labels, test_features, test_labels)
    # print(f'    Neural Network MLP - accuracy: {(round(acc*100, 4))}\n')

    # acc = neural_network_MLP_ours(train_features, train_labels, test_features, test_labels)
    # print(f'    Neural Network MLP ours - accuracy: {(round(acc * 100, 4))}\n')


def KNN(k, train_data, train_labels, test_data, test_labels):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_data, train_labels)
    acc = model.score(test_data, test_labels)
    return acc, k


def SVM(train_data, train_labels, test_data, test_labels, c_of_svm):
    model = SVC(C=c_of_svm, max_iter=1000, class_weight='balanced', gamma='scale')
    model.fit(train_data, train_labels)
    acc = model.score(test_data, test_labels)
    return acc


def neural_network_MLP(train_data, train_labels, test_data, test_labels):
    model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, alpha=1e-4,
                          solver='sgd', tol=1e-4, random_state=1,
                          learning_rate_init=.1)
    model.fit(train_data, train_labels)
    acc = model.score(test_data, test_labels)
    return acc


def neural_network_MLP_ours(train_data, train_labels, test_data, test_labels):
    model = MLPClassifier()
    model.fit(train_data, train_labels)
    acc = model.score(test_data, test_labels)
    return acc


for c_of_svm in [0.2, 0.5, 1, 2]:
    print(f'c_of_svm={c_of_svm}')

    print("[INFO handling images...]")
    rawImages, color_hist, sift_features, set_type, labels = [], [], [], [], []
    # df = pd.read_csv('mini_data_basic_model.csv')
    df = pd.read_csv('data_basic_model_linux.csv')
    df = df.drop(['Unnamed: 0'], axis=1)

    for index, row in df.iterrows():
        path = row['path']

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image is None:
            continue  # TODO right?
        else:
            set_type_ = row['train/test']
            if row['cat/dog'] == 'cat':
                label = 1
                augment_flip_lr, augment_flip_lr_crop, augment_add_noise = \
                    utils.augment_flip_lr(image), utils.augment_flip_lr_crop(image), utils.augment_add_noise(image)
                images = [image, augment_flip_lr, augment_flip_lr_crop, augment_add_noise]
            else:  # row['cat/dog'] == 'dog'
                label = 0
                augment_flip_lr = utils.augment_flip_lr(image)
                images = [image, augment_flip_lr]

            for i in images:
                pixels = image_to_feature_vector(i)
                pixels = pixels/255.0
                rawImages.append(pixels)

                hist = extract_color_histogram(i)
                color_hist.append(hist)

                sift = extract_SIFT(i)
                sift_features.append(sift)

                set_type.append(set_type_)
                labels.append(label)

        # show an update every 1000 images until the last image
        if index > 0 and ((index + 1) % 3000 == 0 or index == len(df) - 1):
            print("[images processed {}/{}]".format(index + 1, len(df)))
            # plt.imshow(image)
            # plt.show()

    print(f'Number of cats after adding augmentations:{sum(labels)}')
    print(f'Number of dogs after adding augmentations:{len(labels) - sum(labels)}\n')

    rawImages, color_hist, labels, set_type, sift_features =\
        np.array(rawImages), np.array(color_hist), np.array(labels), np.array(set_type), np.array(sift_features)

    print(f'color_hist.shape:{color_hist.shape}')
    # print(f'color_hist:{color_hist}')
    print(f'min color_hist:{color_hist.min()}')
    print(f'max color_hist:{color_hist.max()}')
    print(f'rawImages.shape:{rawImages.shape}')
    # print(f'rawImages:{rawImages}')
    print(f'min rawImages:{rawImages.min()}')
    print(f'max rawImages:{rawImages.max()}')
    print(f'sift_features.shape:{sift_features.shape}')
    # print(f'sift_features:{sift_features}')
    # print(f'min sift_features:{sift_features.min()}')
    # print(f'max sift_features:{sift_features.max()}')
    sift_features = sift_features/sift_features.max()
    # print(f'sift_features:{sift_features}')
    print(f'min sift_features:{sift_features.min()}')
    print(f'max sift_features:{sift_features.max()}')

    print()

    print("Evaluating raw pixel accuracy...")
    Classify(rawImages, set_type, labels, c_of_svm)

    print("Evaluating color histogram accuracy...")
    Classify(color_hist, set_type, labels, c_of_svm)

    print("Evaluating SIFT accuracy...")
    Classify(sift_features, set_type, labels, c_of_svm)

    print("Evaluating raw pixels & color histogram accuracy...")
    raw_and_hist = np.concatenate((rawImages, color_hist), axis=1)
    Classify(raw_and_hist, set_type, labels, c_of_svm)

    print("Evaluating raw pixels & color histogram & sift accuracy...")
    raw_and_hist_sift = np.concatenate((np.concatenate((rawImages, color_hist), axis=1), sift_features), axis=1)
    Classify(raw_and_hist_sift, set_type, labels, c_of_svm)
