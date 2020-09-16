"""
Inference phase structured model. we assume 'crfrnn_finetuned_weights.h5' in directory
Use the segmentation from the FCN + CRF-RNN model fine tuned on our VOC, OXFORD datasets
Convert the semantic segmentation into a classification model by extracting the class with the most pixels in an image

Semantic segmentation >> Image classification
"""

import sys
import collections
import random
import pandas as pd
import crfrnn_keras
from crfrnn_keras.crfrnn_model import get_crfrnn_model_def
from sklearn.metrics import confusion_matrix

sys.path.insert(1, './src')


def get_classification_from_segmentation(segmentation_):
    """
    retrieve the classification from the segmentation
    :param segmentation_: a segmentation of the image (pixel values: 0 = background || 8 = cat || 12 = dog)
    :return: 'cat' / 'dog' prediction for the given segmentation
    """
    pixels = list(segmentation_.getdata())
    # create a Counter based on the pixels colors
    counter = collections.Counter(pixels)

    # amount of pixels per category
    cat = counter[8] if 8 in counter else 0
    dog = counter[12] if 12 in counter else 0

    # at the case when there is a tie (unlikely) in the number of dog pixels and cat pixels
    if cat == dog:
        if random.random() < 0.5:
            return 'dog'
        else:
            return 'cat'

    # more cat pixels
    elif cat > dog:
        return 'cat'

    # more dog pixels
    else:
        return 'dog'


if __name__ == '__main__':
    # load the model and the saved weights from train
    finetuned_weights = 'crfrnn_finetuned_weights.h5'

    model = get_crfrnn_model_def()
    model.load_weights(finetuned_weights)

    # get the test data
    df = pd.read_csv('data_basic_model_linux.csv')[['path', 'cat/dog', 'breed', 'dataset', 'train/test']]
    test_data = df[df['train/test'] == 'test']

    predictions = list()
    for index, test_image in test_data.iterrows():
        path = test_image['path']
        img_data, img_h, img_w, size = crfrnn_keras.util.get_preprocessed_image(path)
        probs = model.predict(img_data, verbose=False)[0]
        segmentation = crfrnn_keras.util.get_label_image(probs, img_h, img_w, size)
        predictions.append(get_classification_from_segmentation(segmentation))

    # cm =
    # true \ predicted |  cat   |  dog  |
    #       cat        |   -    |   -   |
    #       dog        |   -    |   -   |
    # Results:
    # (voc + oxford)
    # [[980  20]
    #  [10 990]]
    # accuracy = 98.5%

    cm = confusion_matrix(list(test_data['cat/dog']), predictions, labels=["cat", "dog"])
    print(cm)
