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

# df = pd.read_csv("mini_data_advanced_model.csv")
df = pd.read_csv("data_advanced_model.csv")
df['cat/dog'] = df['cat/dog'].astype(str)
df['breed'] = df['breed'].astype(str)

training_set = df[df['train/test'] == 'train']
full_testing_set = df[df['train/test'] == 'test']

training_set = training_set[['path', 'cat/dog', 'breed']]
full_testing_set = full_testing_set[['path', 'cat/dog', 'breed']]

training_set.reset_index(drop=True, inplace=True)
print(training_set.head())

train_dataGen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_dataGen.flow_from_dataframe(dataframe=training_set, directory="", x_col="path",
                                                    y_col="cat/dog", class_mode="binary", target_size=(224, 224),batch_size=1)

num_of_classes = len(set(training_set['cat/dog']))
animal_model = ResNet50(weights=None, classes=num_of_classes)

# compile the network to initialize the metrics, loss and weights for the network
animal_model.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=['accuracy'])

# Let’s have a look at the description of our CNN
# animal_model.summary()

# train the classifier with the data we gathered by processing the images using ImageDataGenerator class
animal_model.fit(train_generator, epochs=1, steps_per_epoch=5)
# animal_model.fit(train_generator, epochs=30, steps_per_epoch=30)

full_testing_set.reset_index(drop=True, inplace=True)
test_set = pd.DataFrame(full_testing_set['path'])
test_labels = pd.DataFrame(full_testing_set['cat/dog'])
test_labels.reset_index(drop=True, inplace=True)

# Let’s have a look at the unique categories in the training data
classes = train_generator.class_indices

# We will use a reverse of the above dictionary to later convert the predictions to actual classes
inverted_classes = dict(map(reversed, classes.items()))
print(inverted_classes)

# load the images one by one and predict and store the category of each image from the test_set.
Y_pred_prob = []

for i in range(len(test_set)):
    img = image.load_img(path= test_set.path[i],target_size=(224,224,3))
    img = image.img_to_array(img)
    img = img / 255.0  # TODO change?
    test_img = img.reshape((1,224,224,3))
    img_class = animal_model.predict(test_img)
    prediction = img_class[0]
    Y_pred_prob.append(prediction)

Y_pred = tf.argmax(Y_pred_prob, axis=-1).numpy()
Y_pred = [inverted_classes[i] for i in Y_pred]

full_testing_set['animal_prediction'] = Y_pred
print(full_testing_set.head())


dogs_full_testing_set = full_testing_set[full_testing_set['animal_prediction'] == 'dog']
dogs_full_testing_set.reset_index(drop=True, inplace=True)

cats_full_testing_set = full_testing_set[full_testing_set['animal_prediction'] == 'cat']
cats_full_testing_set.reset_index(drop=True, inplace=True)

### dog breed prediction ###
dogs_training_set = training_set[training_set['cat/dog'] == 'dog']

dogs_train_dataGen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
dogs_train_generator = dogs_train_dataGen.flow_from_dataframe(dataframe=dogs_training_set, directory="", x_col="path",
                                                    y_col="breed", class_mode="categorical", target_size=(224, 224),
                                                    batch_size=1)

num_of_classes = len(set(dogs_training_set['breed']))
dogs_model = ResNet50(weights=None, classes=num_of_classes)
dogs_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# dogs_model.summary()
dogs_model.fit(dogs_train_generator, epochs=1, steps_per_epoch=20)
# dogs_model.fit(dogs_train_generator, epochs=30, steps_per_epoch=30)

dogs_test_set = pd.DataFrame(dogs_full_testing_set['path'])
dogs_test_labels = pd.DataFrame(dogs_full_testing_set['breed'])
dogs_test_labels.reset_index(drop=True, inplace=True)

dogs_classes = dogs_train_generator.class_indices
dogs_inverted_classes = dict(map(reversed, dogs_classes.items()))

dogs_Y_pred_prob = []

for i in range(len(dogs_test_set)):
    img = image.load_img(path=dogs_test_set.path[i],target_size=(224,224,3))
    img = image.img_to_array(img)
    img = img / 255.0  # TODO change?
    test_img = img.reshape((1,224,224,3))
    img_class = dogs_model.predict(test_img)
    prediction = img_class[0]
    dogs_Y_pred_prob.append(prediction)

if dogs_Y_pred_prob:
    dogs_Y_pred = tf.argmax(dogs_Y_pred_prob, axis=1).numpy()
    dogs_Y_pred = [dogs_inverted_classes[i] for i in dogs_Y_pred]
    dogs_full_testing_set['breed_prediction'] = dogs_Y_pred

print(dogs_full_testing_set.head())


### cat breed prediction ###

cats_training_set = training_set[training_set['cat/dog'] == 'cat']

cats_train_dataGen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
cats_train_generator = cats_train_dataGen.flow_from_dataframe(dataframe=cats_training_set, directory="", x_col="path",
                                                    y_col="breed", class_mode="categorical", target_size=(224, 224),
                                                    batch_size=1)

num_of_classes = len(set(cats_training_set['breed']))
cats_model = ResNet50(weights=None, classes=num_of_classes)
cats_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# cats_model.summary()
# cats_model.fit(cats_train_generator, epochs=30, steps_per_epoch=30)
cats_model.fit(cats_train_generator, epochs=1, steps_per_epoch=20)


cats_test_set = pd.DataFrame(cats_full_testing_set['path'])
cats_test_labels = pd.DataFrame(cats_full_testing_set['breed'])
cats_test_labels.reset_index(drop=True, inplace=True)

cats_classes = cats_train_generator.class_indices
cats_inverted_classes = dict(map(reversed, cats_classes.items()))


cats_Y_pred_prob = []

for i in range(len(cats_test_set)):
    img = image.load_img(path=cats_test_set.path[i], target_size=(224, 224, 3))
    img = image.img_to_array(img)
    img = img / 255.0  # TODO change?
    test_img = img.reshape((1, 224, 224, 3))
    img_class = cats_model.predict(test_img)
    prediction = img_class[0]
    cats_Y_pred_prob.append(prediction)

if cats_Y_pred_prob:
    cats_Y_pred = tf.argmax(cats_Y_pred_prob, axis=-1).numpy()
    cats_Y_pred = [cats_inverted_classes[i] for i in cats_Y_pred]
    cats_full_testing_set['breed_prediction'] = cats_Y_pred

print(cats_full_testing_set.head())


### accuracies calc ###
concat_df = pd.concat([dogs_full_testing_set, cats_full_testing_set])
print(concat_df.head())

animal_binary_accuracy = len(concat_df[concat_df['cat/dog'] == concat_df['animal_prediction']])/len(concat_df)
print(f'Animal binary accuracy: {animal_binary_accuracy}')

dogs_breed_accuracy = len(concat_df[concat_df['animal_prediction'] == 'dog'][concat_df['breed'] == concat_df['breed_prediction']])/len(concat_df[concat_df['cat/dog']=='dog'])
print(f'Dogs breed accuracy (out of all dogs): {dogs_breed_accuracy}')

cats_breed_accuracy = len(concat_df[concat_df['animal_prediction'] == 'cat'][concat_df['breed'] == concat_df['breed_prediction']])/len(concat_df[concat_df['cat/dog']=='cat'])
print(f'Cats breed accuracy (out of all cats): {cats_breed_accuracy}')

final_accuracy = len(concat_df[concat_df['breed'] == concat_df['breed_prediction']])/len(concat_df)
print(f'Total accuracy (breed prediction): {final_accuracy}')
