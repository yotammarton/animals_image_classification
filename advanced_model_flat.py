from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

TRAIN_BATCH_SIZE = 32
INPUT_SHAPE = [224, 224, 3]  # images will be resized to this shape, this is also the dims for layers

"""LOAD DATAFRAMES"""

df = pd.read_csv("data_advanced_model.csv")
# df = pd.read_csv("mini_data_advanced_model.csv")
df['cat/dog'] = df['cat/dog'].astype(str)
df['breed'] = df['breed'].astype(str)

train_df = df[df['train/test'] == 'train']
test_df = df[df['train/test'] == 'test']
train_df = train_df[['path', 'cat/dog', 'breed']]
test_df = test_df[['path', 'cat/dog', 'breed']]
num_of_classes = len(set(train_df['breed']))

"""CREATE IMAGE GENERATORS"""
train_dataGen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_dataGen.flow_from_dataframe(dataframe=train_df, x_col="path", y_col="breed",
                                                    class_mode="categorical", arget_size=INPUT_SHAPE[:2],
                                                    batch_size=TRAIN_BATCH_SIZE)

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

model = ResNet50(weights=None, classes=num_of_classes)
# print(model.summary())
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print('============ fit model ============')
model.fit(train_generator, epochs=30, steps_per_epoch=np.ceil(len(train_df) / TRAIN_BATCH_SIZE))

print('============ predict model ============')
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
print(f'\nAnimal breed flat accuracy: {accuracy}')

# test_df.to_csv('advanced_flat_model_output_test_df.zip')
