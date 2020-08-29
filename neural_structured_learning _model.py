import tensorflow as tf
import neural_structured_learning as nsl
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow_datasets as tfds
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_INPUT_NAME = 'input_1'  # if experiencing problems with this change to value of 'model.layers[0].name'
LABEL_INPUT_NAME = 'label'
BATCH_SIZE = 1
INPUT_SHAPE = [128, 128, 3]  # images will be resized to this shape, this is also the dims for layers


# TODO choose different shape?


def convert_to_dictionaries(image, label):
    return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}


df = pd.read_csv("data_advanced_model.csv")
df['cat/dog'] = df['cat/dog'].astype(str)
df['breed'] = df['breed'].astype(str)

# df = pd.read_csv("data_advanced_model.csv")
train = df[df['train/test'] == 'train']
test = df[df['train/test'] == 'test']

train = train[['path', 'cat/dog', 'breed']]
test = test[['path', 'cat/dog', 'breed']]
num_of_classes = len(set(train['breed']))


# def generator(train_or_test, df, x_col, y_col, class_mode="categorical"):
def generator(train_or_test):
    """This function is used in order to create the tensorflow.data.Dataset object
    that will read data from dataframe"""
    if train_or_test == 'train':
        train_data_gen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        return train_data_gen.flow_from_dataframe(dataframe=train, x_col="path", y_col="breed",
                                                  class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                  batch_size=BATCH_SIZE)
    else:  # test
        test_data_gen = ImageDataGenerator(rescale=1. / 255)  # without augmentations
        return test_data_gen.flow_from_dataframe(dataframe=test, x_col="path", y_col="breed",
                                                 class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                 batch_size=BATCH_SIZE)


train_dataset = tf.data.Dataset.from_generator(
    generator, args=["train"],
    output_types=(tf.float32, tf.float32),
    output_shapes=([BATCH_SIZE] + INPUT_SHAPE, (BATCH_SIZE, num_of_classes)))

# convert the dataset to the desired format
train_dataset = train_dataset.map(convert_to_dictionaries)

# same for test data
test_dataset = tf.data.Dataset.from_generator(
    generator, args=["test"],
    output_types=(tf.float32, tf.float32),
    output_shapes=([BATCH_SIZE] + INPUT_SHAPE, (BATCH_SIZE, num_of_classes)))
test_dataset = test_dataset.map(convert_to_dictionaries)

model = ResNet50(weights=None, classes=num_of_classes, input_shape=INPUT_SHAPE)

"""############################# NSL #############################"""
adversarial_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.2, adv_grad_norm='infinity')
adversarial_model = nsl.keras.AdversarialRegularization(model, adv_config=adversarial_config)

adversarial_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
adversarial_model.fit(train_dataset, epochs=7, steps_per_epoch=50)

results = adversarial_model.evaluate(test_dataset)  # TODO
print(results)

"""GRAPH SIMILARITY REPRESENTATION - GET THE LAST LAYER'S OUTPUT BEFORE PREDICTIONS"""
# from keras import backend as K
# layer_function = K.function([model.layers[0].input], [model.layers[-2].output])
# # that's the representation of the layer one before the last
# layer_output = layer_function(train_dataset.__iter__().next())[0]

"""
############################################
############################################
############################################
DEAD CODE ISLAND
############################################
############################################
############################################
"""
# def normalize(features):
#     features[IMAGE_INPUT_NAME] = tf.cast(
#         features[IMAGE_INPUT_NAME], dtype=tf.float32) / 255.0
#     return features
#
#
# def convert_to_tuples(features):
#     return features[IMAGE_INPUT_NAME], features[LABEL_INPUT_NAME]

# class HParams(object):
#     def __init__(self):
#         self.input_shape = [128, 128, 3]
#         self.num_classes = 37
#         self.conv_filters = [32, 64, 64]
#         self.kernel_size = (3, 3)
#         self.pool_size = (2, 2)
#         self.num_fc_units = [64]
#         self.batch_size = 1
#         self.epochs = 5
#         self.adv_multiplier = 0.2
#         self.adv_step_size = 0.2
#         self.adv_grad_norm = 'infinity'
#
#
# HPARAMS = HParams()

# datasets = tfds.load('mnist')

# mnist_train_dataset = datasets['train']
# mnist_test_dataset = datasets['test']
#

#
# train_dataset = mnist_train_dataset.map(normalize).shuffle(10000).batch(HPARAMS.batch_size).map(convert_to_tuples)


# test_dataset = test_dataset.map(normalize).batch(HPARAMS.batch_size).map(convert_to_tuples)
#
#
# def build_base_model(hparams):
#     """Builds a model according to the architecture defined in `hparams`."""
#     inputs = tf.keras.Input(
#         shape=hparams.input_shape, dtype=tf.float32, name=IMAGE_INPUT_NAME)
#
#     x = inputs
#     for i, num_filters in enumerate(hparams.conv_filters):
#         x = tf.keras.layers.Conv2D(
#             num_filters, hparams.kernel_size, activation='relu')(
#             x)
#         if i < len(hparams.conv_filters) - 1:
#             # max pooling between convolutional layers
#             x = tf.keras.layers.MaxPooling2D(hparams.pool_size)(x)
#     x = tf.keras.layers.Flatten()(x)
#     for num_units in hparams.num_fc_units:
#         x = tf.keras.layers.Dense(num_units, activation='relu')(x)
#     pred = tf.keras.layers.Dense(hparams.num_classes, activation='softmax')(x)
#     model = tf.keras.Model(inputs=inputs, outputs=pred)
#     return model


#
#
# base_model = build_base_model(HPARAMS)
# base_model.summary()
#
# adv_config = nsl.configs.make_adv_reg_config(
#     multiplier=HPARAMS.adv_multiplier,
#     adv_step_size=HPARAMS.adv_step_size,
#     adv_grad_norm=HPARAMS.adv_grad_norm
# )
#
# base_adv_model = build_base_model(HPARAMS)
# adv_model = nsl.keras.AdversarialRegularization(
#     base_adv_model,
#     label_keys=[LABEL_INPUT_NAME],
#     adv_config=adv_config
# )
#
# train_set_for_adv_model = train_dataset.map(convert_to_dictionaries)
# test_set_for_adv_model = test_dataset.map(convert_to_dictionaries)


# flow_from_dataframe_args = [training_set, None, 'path', 'breed', None, (128, 128),
#                   'rgb', None, 'categorical', 1, True, None, None,
#                   '', 'png', None, 'nearest', True]


# code for renaming layer's name - doesn't help
# for layer in model.layers:
#     # change only the first layer name
#     layer._name = 'image'
#     break
