import argparse
import string
import os
import cv2
import imutils
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


# with tf.io.gfile.GFile("Abyssinian_4.png", 'rb') as fid:
#     encoded_mask_png = fid.read()
# encoded_png_io = io.BytesIO(encoded_mask_png)
# mask = PIL.Image.open(encoded_png_io)
# plt.imshow(np.array(mask) / 2 * 255)
# plt.show()
# python test.py --dataset "set_name" --neighbors "# of neighors"


def image_to_feature_vector(image, size=(100, 100)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(32, 32, 32)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
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


# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []

# directory_name = 'mini_images/'
directory_name = 'bf/'
entries = os.listdir(directory_name)

# loop over the input images
for (i, item) in enumerate(entries):
    if '6' in item:
        path = directory_name + item
        image6 = cv2.imread(path)
        print(f'image6.shape:{image6.shape}')
        window_name = '6'

        gray = cv2.cvtColor(image6, cv2.COLOR_BGR2GRAY)
        print(f'gray6.shape:{gray.shape}')

        sift = cv2.xfeatures2d.SIFT_create(nfeatures=70)
        key_points6, descriptors6 = sift.detectAndCompute(gray, None)  # None is for no-mask
        print(f'key_points6:{key_points6}')
        print(f'key_points6[0]:{key_points6[0]}')
        print(f'key_points6.shape:{len(key_points6)}')
        print(f'descriptors6:{descriptors6}')
        print(f'descriptors6.shape:{descriptors6.shape}')
        print(f'descriptors6.shape:{len(descriptors6.flatten())}')

        sift_image = cv2.drawKeypoints(gray, key_points6, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.waitKey(0)
        cv2.imshow(window_name, sift_image)

        image6 = cv2.imread(path)
        cv2.waitKey(0)
        cv2.imshow('g', sift_image)

    elif '7' in item:
        path = directory_name + item
        image7 = cv2.imread(path)
        print(f'image7.shape:{image7.shape}')
        window_name = '7'

        gray = cv2.cvtColor(image7, cv2.COLOR_BGR2GRAY)
        print(f'gray7.shape:{gray.shape}')

        sift = cv2.xfeatures2d.SIFT_create(nfeatures=70)
        key_points7, descriptors7 = sift.detectAndCompute(gray, None)  # None is for no-mask
        print(f'key_points7:{key_points7}')
        print(f'key_points7.shape:{len(key_points7)}')
        print(f'descriptors7:{descriptors7}')
        print(f'descriptors7.shape:{descriptors7.shape}')
        print(f'descriptors7.shape:{len(descriptors7.flatten())}')
        sift_image = cv2.drawKeypoints(gray, key_points7, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.waitKey(0)
        cv2.imshow(window_name, sift_image)

        image6 = cv2.imread(path)
        cv2.waitKey(0)
        cv2.imshow('g', sift_image)

# keypoints_, descriptors_ = cv2.Feature2D.compute(image7, key_points7, descriptors7)
# print(keypoints_)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# Match descriptors.
# matches = bf.match(descriptors6, descriptors7)
matches = bf.knnMatch(descriptors6,descriptors7, k=2)

# # Sort them in the order of their distance.
# matches = sorted(matches, key=lambda x: x.distance)

good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(image6, key_points6, image7, key_points7, good, None, flags=2)
plt.imshow(img3), plt.show()

# ## regular match and not knn match ###
# # Draw first 10 matches.
# img3 = cv2.drawMatches(image6, key_points6, image7, key_points7, matches[:10], None, flags=2)
# plt.imshow(img3), plt.show()

# for (i, item) in enumerate(entries):
#     if '.jpg' in item:
#         # print(item)
#         path = directory_name + item
#
#         # load the image and extract the class label
#         # our images were named as labels.image_number.format
#         image = cv2.imread(path)  # TODO I think the problem is that it can't read bit depth=8 (and not 24)
#         window_name = 'red'
#         # cv2.waitKey(0)
#         # cv2.imshow(window_name, image)
#
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         sift = cv2.xfeatures2d.SIFT_create()
#         key_points, descriptors = sift.detectAndCompute(gray, None)  # None is for no-mask
#
#         sift_image = cv2.drawKeypoints(gray, key_points, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#         # cv2.imwrite('sift_'+str(item), sift_image)
#
#         cv2.waitKey(0)
#         cv2.imshow(window_name, sift_image)
#
#         cv2.destroyAllWindows()
