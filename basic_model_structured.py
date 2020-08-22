import sys

sys.path.insert(1, './src')
from src.crfrnn_model import get_crfrnn_model_def
from sklearn.metrics import confusion_matrix
import src.util
import collections
import random
import pandas as pd


def get_classification_from_segmentation(segmentation_):
    """
    0 = background || 8 = cat || 12 = dog
    """
    pixels = list(segmentation_.getdata())
    # change all non-background / non-cat / non-dog >> background
    pixels = [p if p in [0, 8, 12] else 0 for p in pixels]
    # create a Counter based on the pixels colors
    counter = collections.Counter(pixels)

    cat = counter[8] if 8 in counter else 0
    dog = counter[12] if 12 in counter else 0

    if cat == dog:
        if random.random() < 0.5:
            return 'dog'
        else:
            return 'cat'
    elif cat > dog:
        return 'cat'
    else:
        return 'dog'


if __name__ == '__main__':
    # load the model and the saved weights from train
    # Download the model from https://goo.gl/ciEYZi  TODO del
    saved_model_path = 'crfrnn_keras_model.h5'

    model = get_crfrnn_model_def()
    model.load_weights(saved_model_path)

    # get the test data
    df = pd.read_csv('data_basic_model_ubuntu.csv')[['path', 'cat/dog', 'breed', 'dataset', 'train/test']]
    test_data = df[df['train/test'] == 'test']

    predictions = list()
    for index, test_image in test_data.iterrows():
        path = test_image['path']
        img_data, img_h, img_w, size = src.util.get_preprocessed_image(path)
        probs = model.predict(img_data, verbose=False)[0]
        segmentation = src.util.get_label_image(probs, img_h, img_w, size)
        predictions.append(get_classification_from_segmentation(segmentation))

    # cm =
    # true \ predicted |  cat   |  dog  |
    #       cat        |   -    |   -   |
    #       dog        |   -    |   -   |

    cm = confusion_matrix(list(test_data['cat/dog']), predictions, labels=["cat", "dog"])
    print(cm)
