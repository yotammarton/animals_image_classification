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


df = pd.read_csv("data_advanced_model.csv")
# df = pd.read_csv("mini_data_advanced_model.csv")
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
                                                    y_col="breed", class_mode="categorical", target_size=(128, 128),
                                                    batch_size=1)  # TODO chgnage batch size?

num_of_classes = len(set(training_set['breed']))
model = ResNet50(weights=None, classes=num_of_classes)  # TODO change weights=None?

# compile the network to initialize the metrics, loss and weights for the network
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy', 'accuracy'])

# Let’s have a look at the description of our CNN
# model.summary()

# train the classifier with the data we gathered by processing the images using ImageDataGenerator class
# model.fit_generator()
# model.fit(train_generator, epochs=50, steps_per_epoch=60)
model.fit(train_generator, epochs=1, steps_per_epoch=20)

test_set = pd.DataFrame(full_testing_set['path'])
test_set.reset_index(drop=True, inplace=True)
test_labels = pd.DataFrame(full_testing_set['breed'])
test_labels.reset_index(drop=True, inplace=True)

# Let’s have a look at the unique categories in the training data
classes = train_generator.class_indices
print(f'classes: {classes}')

# We will use a reverse of the above dictionary to later convert the predictions to actual classes
inverted_classes = dict(map(reversed, classes.items()))
print(f'inverted_classes: {inverted_classes}')

# load the images one by one and predict and store the category of each image from the test_set.
Y_pred = []
for i in range(len(test_set)):
    img = image.load_img(path=test_set.path[i], target_size=(224, 224, 3))
    img = image.img_to_array(img)
    test_img = img.reshape((1, 224, 224, 3))
    img_class = model.predict(test_img)
    prediction = img_class[0]
    Y_pred.append(prediction)

Y_pred_np = np.array(Y_pred)
model_pred, breed_pred = [], []
for i in Y_pred_np:
    model_pred.append(list(i).index(1))
breed_pred = [inverted_classes[i] for i in model_pred]

full_testing_set['predictions'] = breed_pred
accuracy = len(full_testing_set[full_testing_set['breed'] == full_testing_set['predictions']])/len(full_testing_set)
print(f'accuracy: {accuracy}')
