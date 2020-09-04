from tensorflow.keras.applications.resnet50 import ResNet50
# tensorflow.keras.applications.resnet.preprocess_input TODO needed? shown in nightly but not in stable
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from tensorflow.keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from tensorflow.python.keras.applications.efficientnet import EfficientNetB6, EfficientNetB7
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
# from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_dense TODO needed?
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import tensorflow as tf
import pandas as pd
import sys

model_name = sys.argv[1] if len(sys.argv) > 1 else ""
TRAIN_BATCH_SIZE = 32

print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('NEW RUN FOR FLAT MODEL')
print(f'MODEL = {model_name}')

if model_name == 'inception_v3':
    INPUT_SHAPE = [299, 299, 3]
elif model_name == 'efficientnetb7':
    INPUT_SHAPE = [600, 600, 3]
    TRAIN_BATCH_SIZE = 1
elif model_name == 'efficientnetb6':
    INPUT_SHAPE = [528, 528, 3]
elif model_name == 'inception_resnet_v2':
    INPUT_SHAPE = [299, 299, 3]
else:
    INPUT_SHAPE = [224, 224, 3]
# images will be resized to this shape, this is also the dims for layers

"""LOAD DATAFRAMES"""

df = pd.read_csv("data_advanced_model_linux.csv")
# df = pd.read_csv("mini_data_advanced_model.csv")
df['cat/dog'] = df['cat/dog'].astype(str)
df['breed'] = df['breed'].astype(str)

train_df = df[df['train/test'] == 'train'][['path', 'cat/dog', 'breed']]
val_df = df[df['train/test'] == 'validation'][['path', 'cat/dog', 'breed']]
test_df = df[df['train/test'] == 'test'][['path', 'cat/dog', 'breed']]
num_of_classes = len(set(train_df['breed']))

"""CREATE IMAGE GENERATORS"""
if 'efficientnet' in model_name:
    train_dataGen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
elif 'vgg' in model_name:
    pre_process = preprocess_input_vgg16 if '16' in model_name else preprocess_input_vgg19
    train_dataGen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                       preprocessing_function=pre_process)
# elif 'dense' in model_name:
#     pre_process = preprocess_input_dense
#     train_dataGen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
#                                        preprocessing_function=pre_process)
elif model_name == 'inception_resnet_v2':
    pre_process = preprocess_input_inception_resnet_v2
    train_dataGen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                       preprocessing_function=pre_process)
else:
    train_dataGen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_dataGen.flow_from_dataframe(dataframe=train_df, x_col="path", y_col="breed",
                                                    class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                    batch_size=TRAIN_BATCH_SIZE)
if 'efficientnet' in model_name:
    val_data_gen = ImageDataGenerator()
elif 'vgg' in model_name:
    pre_process = preprocess_input_vgg16 if '16' in model_name else preprocess_input_vgg19
    val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
# elif 'dense' in model_name:
#     pre_process = preprocess_input_dense
#     val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
elif model_name == 'inception_resnet_v2':
    pre_process = preprocess_input_inception_resnet_v2
    val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
else:
    val_data_gen = ImageDataGenerator(rescale=1. / 255)  # without augmentations
val_generator = val_data_gen.flow_from_dataframe(dataframe=val_df, x_col="path", y_col="breed",
                                                 class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                 batch_size=1, shuffle=False)

if 'efficientnet' in model_name:
    test_data_gen = ImageDataGenerator()
elif 'vgg' in model_name:
    pre_process = preprocess_input_vgg16 if '16' in model_name else preprocess_input_vgg19
    test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
# elif 'dense' in model_name:
#     pre_process = preprocess_input_dense
#     test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
elif model_name == 'inception_resnet_v2':
    pre_process = preprocess_input_inception_resnet_v2
    test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
else:
    test_data_gen = ImageDataGenerator(rescale=1. / 255)  # without augmentations
test_generator = test_data_gen.flow_from_dataframe(dataframe=test_df, x_col="path", y_col="breed",
                                                   class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                   batch_size=1, shuffle=False)  # batch_size=1, shuffle=False for test!

"""PREPARE TENSORFLOW DATASETS FOR TEST"""
test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_types=(tf.float32, tf.float32))
# for test data we dont want to generate infinite data, we just want the amount of data in the test (that's why take())
test_dataset = test_dataset.take(len(test_df))  # Note: test_generator must have shuffle=False

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_generator,
    output_types=(tf.float32, tf.float32))
val_dataset = val_dataset.take(len(val_df))

if model_name == 'resnet50':
    model = ResNet50(weights=None, classes=num_of_classes)
if model_name == 'resnet50v2':
    model = ResNet50V2(weights=None, classes=num_of_classes)
elif model_name == 'vgg16':
    model = VGG16(weights=None, classes=num_of_classes)
elif model_name == 'vgg19':
    model = VGG19(weights=None, classes=num_of_classes)
elif model_name == 'inception_v3':
    model = InceptionV3(weights=None, classes=num_of_classes)
elif model_name == 'inception_resnet_v2':
    model = InceptionResNetV2(weights=None, classes=num_of_classes)
elif model_name == 'efficientnetb7':
    model = EfficientNetB7(weights=None, classes=num_of_classes)
elif model_name == 'efficientnetb6':
    model = EfficientNetB6(weights=None, classes=num_of_classes)
elif model_name == 'densenet121':
    model = DenseNet121(weights=None, classes=num_of_classes)
elif model_name == 'densenet169':
    model = DenseNet169(weights=None, classes=num_of_classes)
elif model_name == 'densenet201':
    model = DenseNet201(weights=None, classes=num_of_classes)
else:
    raise ValueError("not supported model name")

# print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath=f'flat_weights_{model_name}.hdf5', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
# TODO monitor = 'val_accuracy' / 'val_loss'

print('============ fit flat model ============')
model.fit(train_generator, epochs=1, steps_per_epoch=np.ceil(len(train_df) / TRAIN_BATCH_SIZE),
          validation_data=val_dataset, callbacks=[checkpoint, early_stopping])
model.load_weights(filepath=f'flat_weights_{model_name}.hdf5')

print('============ predict flat model ============')
# Let’s have a look at the unique categories in the training data
classes = train_generator.class_indices
# We will use a reverse of the above dictionary to later convert the predictions to actual classes
inverted_classes = dict(map(reversed, classes.items()))
print(inverted_classes)
predictions = model.predict(test_dataset)
predictions = tf.argmax(predictions, axis=-1).numpy()
inverted_class_predictions = [inverted_classes[i] for i in predictions]

test_df['flat_prediction'] = inverted_class_predictions
print(test_df)

accuracy = len(test_df[test_df['breed'] == test_df['flat_prediction']]) / len(test_df)
print(f'\n#RESULTS {model_name}# Flat Animal breed accuracy: {accuracy}')

# test_df.to_csv('advanced_flat_model_output_test_df.zip')
