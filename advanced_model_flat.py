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

# df = pd.read_csv("data_advanced_model.csv")
df = pd.read_csv("mini_data_advanced_model.csv")
df['cat/dog'] = df['cat/dog'].astype(str)
df['breed'] = df['breed'].astype(str)

# df = pd.read_csv("data_advanced_model.csv")
training_set = df[df['train/test'] == 'train']
full_testing_set = df[df['train/test'] == 'test']

training_set = training_set[['path', 'cat/dog', 'breed']]
full_testing_set = full_testing_set[['path', 'cat/dog', 'breed']]

print(training_set.head())

train_dataGen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_dataGen.flow_from_dataframe(dataframe=training_set, directory="", x_col="path",
                                                    y_col="breed", class_mode="categorical", target_size=(128, 128),
                                                    batch_size=1)

num_of_classes = len(set(training_set['breed']))
model = ResNet50(weights=None, classes=num_of_classes)

# compile the network to initialize the metrics, loss and weights for the network
# model.compile()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy', 'accuracy'])

# Let’s have a look at the description of our CNN
model.summary()

# train the classifier with the data we gathered by processing the images using ImageDataGenerator class
# model.fit_generator()
model.fit_generator(train_generator, epochs=1, steps_per_epoch=50)
# model.fit()
test_set = pd.DataFrame(full_testing_set['path'])
print(test_set.head())

# Let’s have a look at the unique categories in the training data
classes = train_generator.class_indices
print(classes)

# We will use a reverse of the above dictionary to later convert the predictions to actual classes
inverted_classes = dict(map(reversed, classes.items()))
print(inverted_classes)

# load the images one by one and predict and store the category of each image from the test_set.
from tensorflow.keras.preprocessing import image

Y_pred = []
for i in range(len(test_set)):
  img = image.load_img(path= test_set.path[i],target_size=(256,256,3))
  img = image.img_to_array(img)
  test_img = img.reshape((1,256,256,3))
  img_class = model.predict_classes(test_img)
  # img_class = model.call(test_img)
  img_class = model.precict(test_img)
  prediction = img_class[0]
  Y_pred.append(prediction)




##########
# img_path = 'images\miniature_pinscher_79.jpg'
# img = image.load_img(img_path, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# preds = model.predict(x)
# # decode the results into a list of tuples (class, description, probability)
# # (one such list for each sample in the batch)
# print('Predicted:', decode_predictions(preds, top=3)[0])
#
