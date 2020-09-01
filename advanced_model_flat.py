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
                                                    y_col="breed", class_mode="categorical", target_size=(224, 224),
                                                    batch_size=32)

num_of_classes = len(set(training_set['breed']))
model = ResNet50(weights=None, classes=num_of_classes)

# compile the network to initialize the metrics, loss and weights for the network
# model.compile()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy', 'accuracy'])

# Let’s have a look at the description of our CNN
# model.summary()

# train the classifier with the data we gathered by processing the images using ImageDataGenerator class
# model.fit(train_generator, epochs=1, steps_per_epoch=20)
model.fit(train_generator, epochs=50, steps_per_epoch=60)  # TODO what does the prints mean? what is the accuracy, train accuracy?

full_testing_set.reset_index(drop=True, inplace=True)
test_set = pd.DataFrame(full_testing_set['path'])
test_labels = pd.DataFrame(full_testing_set['breed'])
test_labels.reset_index(drop=True, inplace=True)

# Let’s have a look at the unique categories in the training data
classes = train_generator.class_indices

# We will use a reverse of the above dictionary to later convert the predictions to actual classes
inverted_classes = dict(map(reversed, classes.items()))
print(inverted_classes)

Y_pred_prob = []

for i in range(len(test_set)):
    img = image.load_img(path=test_set.path[i], target_size=(224, 224, 3))
    img = image.img_to_array(img)
    img = img / 255.0  # TODO change?
    test_img = img.reshape((1, 224, 224, 3))
    img_class = model.predict(test_img)
    prediction = img_class[0]
    Y_pred_prob.append(prediction)

if Y_pred_prob:
    Y_pred = tf.argmax(Y_pred_prob, axis=-1).numpy()
    Y_pred = [inverted_classes[i] for i in Y_pred]

    full_testing_set['flat prediction'] = Y_pred
    print(full_testing_set)


accuracy = len(full_testing_set[full_testing_set['breed'] == full_testing_set['flat prediction']])/len(full_testing_set)
print()
print(f'Animal breed flat accuracy: {accuracy}')

# full_testing_set.to_csv('advanced_flat_model_output_test_df.zip')
