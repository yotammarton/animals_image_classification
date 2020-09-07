"""
29/8/2020
This module uses adversarial regularization from NSL (neural structured learning)
to predict the breed (flat) of the animal (a multi-class classification problem from 37 different classes)
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import neural_structured_learning as nsl
import pandas as pd
import numpy as np
import sys

IMAGE_INPUT_NAME = 'input_1'  # if experiencing problems with this change to value of 'model.layers[0].name'
LABEL_INPUT_NAME = 'label'

INPUT_SHAPE = [299, 299, 3]
model_name = sys.argv[1] if len(sys.argv) > 1 else ""
TRAIN_BATCH_SIZE = 16
multiplier, adv_step_size, adv_grad_norm = float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]

print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('NEW RUN FOR NSL MODEL')
print(f'MODEL = {model_name}, BATCH_SIZE = {TRAIN_BATCH_SIZE}')
print(f'multiplier = {multiplier},  adv_step_size = {adv_step_size}, adv_grad_norm = {adv_grad_norm}')


def convert_to_dictionaries(image, label):
    return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}


"""LOAD DATAFRAMES"""
df = pd.read_csv("data_advanced_model_linux.csv")
df['cat/dog'] = df['cat/dog'].astype(str)
df['breed'] = df['breed'].astype(str)

train_df = df[df['train/test'] == 'train'][['path', 'cat/dog', 'breed']]
val_df = df[df['train/test'] == 'validation'][['path', 'cat/dog', 'breed']]
test_df = df[df['train/test'] == 'test'][['path', 'cat/dog', 'breed']]
num_of_classes = len(set(train_df['breed']))

"""CREATE IMAGE GENERATORS"""
# TODO only good for inception_resnet_v2
pre_process = preprocess_input_inception_resnet_v2
train_data_gen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                    preprocessing_function=pre_process)

train_generator = train_data_gen.flow_from_dataframe(dataframe=train_df, x_col="path", y_col="breed",
                                                     class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                     batch_size=TRAIN_BATCH_SIZE)

val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
val_generator = val_data_gen.flow_from_dataframe(dataframe=val_df, x_col="path", y_col="breed",
                                                 class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                 batch_size=1, shuffle=False)

test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
test_generator = test_data_gen.flow_from_dataframe(dataframe=test_df, x_col="path", y_col="breed",
                                                   class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                   batch_size=1, shuffle=False)

"""PREPARE TENSORFLOW DATASETS FOR TRAIN, TEST"""
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32))
# convert the dataset to the desired format of NSL (dictionaries)
train_dataset = train_dataset.map(convert_to_dictionaries)

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_generator,
    output_types=(tf.float32, tf.float32))
val_dataset = val_dataset.map(convert_to_dictionaries)
val_dataset = val_dataset.take(len(val_df))

# same for test data
test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_types=(tf.float32, tf.float32))
test_dataset = test_dataset.map(convert_to_dictionaries)
# for test data we dont want to generate infinite data, we just want the amount of data in the test (that's why take())
test_dataset = test_dataset.take(len(test_df))  # Note: test_generator must have shuffle=False

"""DEFINE BASE MODEL"""
model = InceptionResNetV2(weights=None, classes=num_of_classes)

"""NSL"""
# TODO play with parm of adversarial_config
adversarial_config = nsl.configs.make_adv_reg_config(multiplier=multiplier,
                                                     adv_step_size=adv_step_size,
                                                     adv_grad_norm=adv_grad_norm)
adversarial_model = nsl.keras.AdversarialRegularization(model, label_keys=[LABEL_INPUT_NAME],
                                                        adv_config=adversarial_config)

checkpoint = ModelCheckpoint(filepath=f'nsl_weights_{model_name}_{multiplier}_{adv_step_size}_{adv_grad_norm}.hdf5',
                             save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(patience=5, verbose=1)
adversarial_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# TODO different loss?
print('============ fit adversarial model ============')
adversarial_model.fit(train_dataset, epochs=100, steps_per_epoch=np.ceil(len(train_df) / TRAIN_BATCH_SIZE),
                      validation_data=val_dataset, callbacks=[checkpoint, early_stopping, reduce_lr])

adversarial_model.load_weights(filepath=f'nsl_weights_{model_name}_{multiplier}_{adv_step_size}_{adv_grad_norm}.hdf5')

print('================== inference ==================')
# predictions = adversarial_model.predict(test_dataset)  # TODO
result = adversarial_model.evaluate(test_dataset)
print(f'#RESULTS# NSL model: {dict(zip(adversarial_model.metrics_names, result))}'
      f'model_name: {model_name}\n'
      f'multiplier: {multiplier}\n'
      f'adv_step_size: {adv_step_size}'
      f'adv_grad_norm: {adv_grad_norm}')

"""########################"""

#
# def simple_model():
#     class HParams(object):
#         def __init__(self):
#             self.input_shape = INPUT_SHAPE
#             self.num_classes = 37
#             self.conv_filters = [32, 64, 64]
#             self.kernel_size = (3, 3)
#             self.pool_size = (2, 2)
#             self.num_fc_units = [64]
#             self.batch_size = 32
#             self.epochs = 30
#             self.adv_multiplier = 0.2
#             self.adv_step_size = 0.2
#             self.adv_grad_norm = 'infinity'
#
#     HPARAMS = HParams()
#
#     def build_base_model(hparams):
#         """Builds a model according to the architecture defined in `hparams`."""
#         inputs = tf.keras.Input(
#             shape=hparams.input_shape, dtype=tf.float32, name=IMAGE_INPUT_NAME)
#
#         x = inputs
#         for i, num_filters in enumerate(hparams.conv_filters):
#             x = tf.keras.layers.Conv2D(
#                 num_filters, hparams.kernel_size, activation='relu')(
#                 x)
#             if i < len(hparams.conv_filters) - 1:
#                 # max pooling between convolutional layers
#                 x = tf.keras.layers.MaxPooling2D(hparams.pool_size)(x)
#         x = tf.keras.layers.Flatten()(x)
#         for num_units in hparams.num_fc_units:
#             x = tf.keras.layers.Dense(num_units, activation='relu')(x)
#         pred = tf.keras.layers.Dense(hparams.num_classes, activation='softmax')(x)
#         model = tf.keras.Model(inputs=inputs, outputs=pred)
#         return model
#
#     base_model = build_base_model(HPARAMS)
#
#     base_model.compile(optimizer='adam', loss='categorical_crossentropy',
#                        metrics=['acc'])
#     """try simple model"""
#     print('================== fit simple model ==================')
#     print(base_model.summary())
#     base_model.fit(train_generator, epochs=30, steps_per_epoch=np.ceil(len(train_df) / TRAIN_BATCH_SIZE))
#     print('================== inference ==================')
#     result = base_model.evaluate(test_dataset)
#     print(dict(zip(base_model.metrics_names, result)))
#

"""
############################################
############################################
############################################
DEAD CODE ISLAND
############################################
############################################
############################################
"""
"""GRAPH SIMILARITY REPRESENTATION - GET THE LAST LAYER'S OUTPUT BEFORE PREDICTIONS"""
# from keras import backend as K
# layer_function = K.function([model.layers[0].input], [model.layers[-2].output])
# # that's the representation of the layer one before the last
# layer_output = layer_function(train_dataset.__iter__().next())[0]

# def normalize(features):
#     features[IMAGE_INPUT_NAME] = tf.cast(
#         features[IMAGE_INPUT_NAME], dtype=tf.float32) / 255.0
#     return features
#
#
# def convert_to_tuples(features):
#     return features[IMAGE_INPUT_NAME], features[LABEL_INPUT_NAME]

# datasets = tfds.load('mnist')

# mnist_train_dataset = datasets['train']
# mnist_test_dataset = datasets['test']
#

# train_dataset = mnist_train_dataset.map(normalize).shuffle(10000).batch(HPARAMS.batch_size).map(convert_to_tuples)

# test_dataset = mnist_test_dataset.map(normalize).batch(BATCH_SIZE).map(convert_to_tuples)

#

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

# flow_from_dataframe_args = [train_df, None, 'path', 'breed', None, (128, 128),
#                   'rgb', None, 'categorical', 1, True, None, None,
#                   '', 'png', None, 'nearest', True]

# code for renaming layer's name - doesn't help
# for layer in model.layers:
#     # change only the first layer name
#     layer._name = 'image'
#     break
