from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import pandas as pd
import sys

model_name = sys.argv[1] if len(sys.argv) > 1 else ""  # TODO
# model_name = '' # choose your own model

print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('NEW RUN FOR HIERARCHICAL MODEL')
print(f'MODEL = {model_name}')

TRAIN_BATCH_SIZE = 32
INPUT_SHAPE = [299, 299, 3] if model_name == 'inception_v3' else [224, 224, 3]
# images will be resized to this shape, this is also the dims for layers

"""LOAD DATAFRAMES"""
df = pd.read_csv("data_advanced_model_linux.csv")
# df = pd.read_csv("mini_data_advanced_model.csv")
df['cat/dog'] = df['cat/dog'].astype(str)
df['breed'] = df['breed'].astype(str)

train_df = df[df['train/test'] == 'train']
test_df = df[df['train/test'] == 'test']
train_df = train_df[['path', 'cat/dog', 'breed']]
test_df = test_df[['path', 'cat/dog', 'breed']]
num_of_classes = len(set(train_df['cat/dog']))

dogs_train_df = train_df[train_df['cat/dog'] == 'dog']
dogs_test_df = test_df[test_df['cat/dog'] == 'dog']
dogs_num_of_classes = len(set(dogs_train_df['breed']))

cats_train_df = train_df[train_df['cat/dog'] == 'cat']
cats_test_df = test_df[test_df['cat/dog'] == 'cat']
cats_num_of_classes = len(set(cats_train_df['breed']))

"""CREATE IMAGE GENERATORS"""
# binary model #
train_dataGen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_dataGen.flow_from_dataframe(dataframe=train_df, x_col="path", y_col="cat/dog",
                                                    class_mode="categorical", arget_size=INPUT_SHAPE[:2],
                                                    batch_size=TRAIN_BATCH_SIZE)

test_data_gen = ImageDataGenerator(rescale=1. / 255)  # without augmentations
test_generator = test_data_gen.flow_from_dataframe(dataframe=test_df, x_col="path", y_col="cat/dog",
                                                   class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                   batch_size=1, shuffle=False)  # batch_size=1, shuffle=False for test!

# dogs model #
dogs_train_dataGen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
dogs_train_generator = dogs_train_dataGen.flow_from_dataframe(dataframe=dogs_train_df, x_col="path", y_col="breed",
                                                              class_mode="categorical", arget_size=INPUT_SHAPE[:2],
                                                              batch_size=TRAIN_BATCH_SIZE)

dogs_test_data_gen = ImageDataGenerator(rescale=1. / 255)  # without augmentations
dogs_test_generator = dogs_test_data_gen.flow_from_dataframe(dataframe=dogs_test_df, x_col="path", y_col="breed",
                                                             class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                             batch_size=1,
                                                             shuffle=False)  # batch_size=1, shuffle=False for test!

# cats model #
cats_train_dataGen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
cats_train_generator = cats_train_dataGen.flow_from_dataframe(dataframe=cats_train_df, x_col="path", y_col="breed",
                                                              class_mode="categorical", arget_size=INPUT_SHAPE[:2],
                                                              batch_size=TRAIN_BATCH_SIZE)

cats_test_data_gen = ImageDataGenerator(rescale=1. / 255)  # without augmentations
cats_test_generator = cats_test_data_gen.flow_from_dataframe(dataframe=cats_test_df, x_col="path", y_col="breed",
                                                             class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                             batch_size=1,
                                                             shuffle=False)  # batch_size=1, shuffle=False for test!

"""PREPARE TENSORFLOW DATASETS FOR TEST"""
# binary model #
test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_types=(tf.float32, tf.float32))
# for test data we dont want to generate infinite data, we just want the amount of data in the test (that's why take())
test_dataset = test_dataset.take(len(test_df))  # Note: test_generator must have shuffle=False

# dogs model #
dogs_test_dataset = tf.data.Dataset.from_generator(
    lambda: dogs_test_generator,
    output_types=(tf.float32, tf.float32))
# for test data we dont want to generate infinite data, we just want the amount of data in the test (that's why take())
dogs_test_dataset = dogs_test_dataset.take(len(dogs_test_df))  # Note: test_generator must have shuffle=False

# cats model #
cats_test_dataset = tf.data.Dataset.from_generator(
    lambda: cats_test_generator,
    output_types=(tf.float32, tf.float32))
# for test data we dont want to generate infinite data, we just want the amount of data in the test (that's why take())
cats_test_dataset = cats_test_dataset.take(len(cats_test_df))  # Note: test_generator must have shuffle=False

"""TRAIN MODELS"""
# binary model #
if model_name == 'resnet50':
    binary_model = ResNet50(weights=None, classes=num_of_classes)
elif model_name == 'vgg16':
    binary_model = VGG16(weights=None, classes=num_of_classes)
elif model_name == 'vgg19':
    binary_model = VGG19(weights=None, classes=num_of_classes)
elif model_name == 'inception_v3':
    binary_model = InceptionV3(weights=None, classes=num_of_classes)
elif model_name == 'efficientnetb7':
    binary_model = EfficientNetB7(weights=None, classes=num_of_classes)
else:
    raise ValueError("not supported model name")

# print(model.summary())
binary_model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])
print('============ binary model fit ============')
binary_model.fit(train_generator, epochs=20, steps_per_epoch=np.ceil(len(train_df) / TRAIN_BATCH_SIZE))

# dogs model #
if model_name == 'resnet50':
    dogs_model = ResNet50(weights=None, classes=dogs_num_of_classes)
elif model_name == 'vgg16':
    dogs_model = VGG16(weights=None, classes=dogs_num_of_classes)
elif model_name == 'vgg19':
    dogs_model = VGG19(weights=None, classes=dogs_num_of_classes)
elif model_name == 'inception_v3':
    dogs_model = InceptionV3(weights=None, classes=dogs_num_of_classes)
elif model_name == 'efficientnetb7':
    dogs_model = EfficientNetB7(weights=None, classes=dogs_num_of_classes)
else:
    raise ValueError("not supported model name")

# print(dogs_model.summary())
dogs_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('============ dogs model fit ============')
dogs_model.fit(dogs_train_generator, epochs=20, steps_per_epoch=np.ceil(len(dogs_train_df) / TRAIN_BATCH_SIZE))

# cats model #
if model_name == 'resnet50':
    cats_model = ResNet50(weights=None, classes=cats_num_of_classes)
elif model_name == 'vgg16':
    cats_model = VGG16(weights=None, classes=cats_num_of_classes)
elif model_name == 'vgg19':
    cats_model = VGG19(weights=None, classes=cats_num_of_classes)
elif model_name == 'inception_v3':
    cats_model = InceptionV3(weights=None, classes=cats_num_of_classes)
elif model_name == 'efficientnetb7':
    cats_model = EfficientNetB7(weights=None, classes=cats_num_of_classes)
else:
    raise ValueError("not supported model name")

# print(cats_model.summary())
cats_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('============ cats model fit ============')
cats_model.fit(cats_train_generator, epochs=20, steps_per_epoch=np.ceil(len(cats_train_df) / TRAIN_BATCH_SIZE))

"""EVALUATE MODELS"""
print('============ binary model evaluate ============')
# binary model #
binary_result = binary_model.evaluate(test_dataset)
print(dict(zip(binary_model.metrics_names, binary_result)))

# dogs model #
print('============ dogs model evaluate ============')
dogs_result = dogs_model.evaluate(dogs_test_dataset)
print(dict(zip(dogs_model.metrics_names, dogs_result)))

# cats model #
print('============ cats model evaluate ============')
cats_result = cats_model.evaluate(cats_test_dataset)
print(dict(zip(cats_model.metrics_names, cats_result)))

"""PREDICT HIERARCHICAL PIPELINE"""
# binary prediction #
classes = train_generator.class_indices
inverted_classes = dict(map(reversed, classes.items()))
print(inverted_classes)
binary_predictions = binary_model.predict(test_dataset)
binary_predictions = tf.argmax(binary_predictions, axis=-1).numpy()
binary_predictions = [inverted_classes[i] for i in binary_predictions]

test_df['binary_prediction'] = binary_predictions
print(test_df)

# dog breed prediction #
predicted_as_dogs_df = test_df[test_df['binary_prediction'] == 'dog']
dogs_classes = dogs_train_generator.class_indices
dogs_inverted_classes = dict(map(reversed, dogs_classes.items()))
print(dogs_inverted_classes)

predicted_as_dogs_data_gen = ImageDataGenerator(rescale=1. / 255)  # without augmentations
predicted_as_dogs_generator = predicted_as_dogs_data_gen.flow_from_dataframe(dataframe=predicted_as_dogs_df,
                                                                             x_col="path",
                                                                             y_col="breed",
                                                                             class_mode="categorical",
                                                                             target_size=INPUT_SHAPE[:2],
                                                                             batch_size=1,
                                                                             shuffle=False)

predicted_as_dogs_dataset = tf.data.Dataset.from_generator(
    lambda: predicted_as_dogs_generator,
    output_types=(tf.float32, tf.float32))
# for test data we dont want to generate infinite data, we just want the amount of data in the test (that's why take())
predicted_as_dogs_dataset = predicted_as_dogs_dataset.take(len(predicted_as_dogs_df))

dogs_predictions = dogs_model.predict(predicted_as_dogs_dataset)
dogs_predictions = tf.argmax(dogs_predictions, axis=-1).numpy()
dogs_predictions = [dogs_inverted_classes[i] for i in dogs_predictions]

predicted_as_dogs_df['breed_prediction'] = dogs_predictions
print(predicted_as_dogs_df)

# cat breed prediction #
predicted_as_cats_df = test_df[test_df['binary_prediction'] == 'cat']
cats_classes = cats_train_generator.class_indices
cats_inverted_classes = dict(map(reversed, cats_classes.items()))
print(cats_inverted_classes)

predicted_as_cats_data_gen = ImageDataGenerator(rescale=1. / 255)  # without augmentations
predicted_as_cats_generator = predicted_as_cats_data_gen.flow_from_dataframe(dataframe=predicted_as_cats_df,
                                                                             x_col="path",
                                                                             y_col="breed",
                                                                             class_mode="categorical",
                                                                             target_size=INPUT_SHAPE[:2],
                                                                             batch_size=1,
                                                                             shuffle=False)

predicted_as_cats_dataset = tf.data.Dataset.from_generator(
    lambda: predicted_as_cats_generator,
    output_types=(tf.float32, tf.float32))
# for test data we dont want to generate infinite data, we just want the amount of data in the test (that's why take())
predicted_as_cats_dataset = predicted_as_cats_dataset.take(len(predicted_as_cats_df))

cats_predictions = cats_model.predict(predicted_as_cats_dataset)
cats_predictions = tf.argmax(cats_predictions, axis=-1).numpy()
cats_predictions = [cats_inverted_classes[i] for i in cats_predictions]

predicted_as_cats_df['breed_prediction'] = cats_predictions
print(predicted_as_cats_df)

"""RESULTS CALCULATION"""

concat_df = pd.concat([predicted_as_dogs_df, predicted_as_cats_df])
print(concat_df.head())

binary_accuracy = len(test_df[test_df['cat/dog'] == test_df['binary_prediction']]) / len(test_df)
print(f'Hierarchical animal binary accuracy: {binary_accuracy}')

final_accuracy = len(concat_df[concat_df['breed'] == concat_df['breed_prediction']]) / len(concat_df)
print(f'Hierarchical breed accuracy (out of 37): {final_accuracy}')

# code falls here TODO
dogs_breed_accuracy = len(
    predicted_as_dogs_df[predicted_as_dogs_df['breed'] == predicted_as_dogs_df['breed_prediction']]) / len(
    dogs_test_dataset)
print(f'Hierarchical dogs breed accuracy (out of 25): {dogs_breed_accuracy}')

cats_breed_accuracy = len(
    predicted_as_cats_df[predicted_as_cats_df['breed'] == predicted_as_cats_df['breed_prediction']]) / len(
    cats_test_dataset)
print(f'Hierarchical cats breed accuracy (out of 12): {cats_breed_accuracy}')
