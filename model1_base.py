import argparse
import string
import os
import cv2
import imutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# path = r'images\miniature_pinscher_163.jpg'
# image = cv2.imread(path)
# image = plt.imread(path)
# image = mpimg.imread(path)

# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# cv2.imshow('red', image)
# cv2.waitKey(0)

# plt.imshow(image)

def image_to_feature_vector(image, size=(100, 100)):
    # resize the image to a fixed size, then flatten the image into a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(32, 32, 32)):
    # extract a 3D color histogram from the HSV color space using the supplied number of `bins` per channel
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])

    # handle normalizing the histogram if we are using OpenCV 2.4.X
    if imutils.is_cv2():
        hist = cv2.normalize(hist)

    # otherwise, perform "in place" normalization in OpenCV 3
    else:
        cv2.normalize(hist, hist)

    # return the flattened histogram as the feature vector
    return hist.flatten()


def extract_SIFT(image, num_of_features=70):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_of_features)

        key_points, descriptors = sift.detectAndCompute(gray, None)  # None is for no-mask
        if len(key_points) != num_of_features:
            key_points = key_points[:num_of_features]
            descriptors = descriptors[:num_of_features]

        # print(f'descriptors6.shape:{len(descriptors.flatten())}')
        sift_image = cv2.drawKeypoints(gray, key_points, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('r', sift_image)
        # cv2.waitKey(0)
        return descriptors.flatten()


print("[INFO] handling images...")
# initialize the raw pixel intensities matrix, the features matrix, and labels list
rawImages, color_hist, sift_features, labels = [], [], [], []

df = pd.read_csv('mini_paths.csv')
df = df.drop(['Unnamed: 0'], axis=1)
print(df)

for index, row in df.iterrows():
    path = row['path']

    # load the image and extract the class label
    image = cv2.imread(path)  # TODO I think the problem is that it can't read bit depth=8 (and not 24)
    # if type(image) == NoneType:
    if image is None:
        print(path)
    else:
        # get the labels from the name of the images by extract the string before "."
        label = 1 if row['cat/dog'] == 'cat' else 0  # 1=cat, 0=dog

        pixels = image_to_feature_vector(image)
        hist = extract_color_histogram(image)
        sift = extract_SIFT(image, 70)

        # add the messages we got to the raw images, features, and labels matricies
        rawImages.append(pixels)
        color_hist.append(hist)
        sift_features.append(sift)
        labels.append(label)

    # show an update every 200 images until the last image
    if index > 0 and ((index + 1) % 2000 == 0 or index == len(df) - 1):
        print("[INFO] processed {}/{}".format(index + 1, len(df)))

print(f'Number of cats:{sum(labels)}')
print(f'Number of dogs:{len(labels) - sum(labels)}\n')

rawImages = np.array(rawImages)
color_hist = np.array(color_hist)
labels = np.array(labels)
sift_features = np.array(sift_features)
print(f'Raw images features: {rawImages}')
print(f'Color histogram features: {color_hist}')
print(f'SIFT features: {sift_features}')

print(f'labels: {labels}')
print(color_hist.shape)
print(rawImages.shape)
print(sift_features.shape)


## memory info ##
# print("[INFO] pixels matrix: {:.2f}MB".format(
#     rawImages.nbytes / (1024 * 1000.0)))
# print("[INFO] features matrix: {:.2f}MB".format(
#     features.nbytes / (1024 * 1000.0)))


def Classify(features, labels):
    # partition the data into training and testing splits, using 85% of the data for training and the remaining 15% for testing
    print(f'    features shape: {features.shape}')

    (train_features, test_features, train_labels, test_labels) = train_test_split(features, labels, test_size=0.15, random_state=777)
    acc, k = KNN(3, train_features, train_labels, test_features, test_labels)
    print(f'    {k}NN - accuracy: {(round(acc*100, 4))}')

    acc = SVM(train_features, train_labels, test_features, test_labels)
    print(f'    SVM - accuracy: {(round(acc*100, 4))}\n')


def KNN(k, train_data, train_labels, test_data, test_labels):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_data, train_labels)
    acc = model.score(test_data, test_labels)
    return acc, k


def SVM(train_data, train_labels, test_data, test_labels):
    model = SVC(max_iter=1000, class_weight='balanced', gamma='scale')
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


print("Evaluating raw pixel accuracy...")
Classify(rawImages, labels)

print("Evaluating color histogram accuracy...")
Classify(color_hist, labels)

print("Evaluating SIFT accuracy...")
Classify(sift_features, labels)

# print("Evaluating raw pixels & color histogram accuracy...")
# raw_and_hist = np.concatenate((rawImages, color_hist), axis=1)
# Classify(raw_and_hist, labels)

print("Evaluating raw pixels & color histogram & sift accuracy...")
raw_and_hist_sift = np.concatenate((np.concatenate((rawImages, color_hist), axis=1), sift_features), axis=1)
Classify(raw_and_hist_sift, labels)


