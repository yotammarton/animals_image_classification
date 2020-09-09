"""
This code use 2 models in order to show the power of adversarial regularization model
trained as part of the NSL framework
1. 'base' model -
2. 'adversarial' model

We load the weights we trained for both models,
And perturb examples to show the power of the adversarial model
TODO
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
import neural_structured_learning as nsl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras
import sys

IMAGE_INPUT_NAME = 'input_1'  # if experiencing problems with this change to value of 'model.layers[0].name'
LABEL_INPUT_NAME = 'label'

INPUT_SHAPE = [299, 299, 3]
SEED = 53
BATCH_SIZE = 128  # TODO
NUMBER_OF_BATCHES = 1  # TODO
multiplier, adv_step_size, adv_grad_norm = 0.2, 0.2, 'l2'


def convert_to_dictionaries(image, label):
    return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}


"""LOAD DATAFRAMES"""
df = pd.read_csv("data_advanced_model_linux.csv")
df['cat/dog'] = df['cat/dog'].astype(str)
df['breed'] = df['breed'].astype(str)

test_df = df[df['train/test'] == 'test'][['path', 'cat/dog', 'breed']]
num_of_classes = len(set(test_df['breed']))

pre_process = preprocess_input_inception_resnet_v2
test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
test_generator_1 = test_data_gen.flow_from_dataframe(dataframe=test_df, x_col="path", y_col="breed",
                                                     class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                     batch_size=BATCH_SIZE, shuffle=True, seed=SEED)  # TODO
print(test_generator_1.class_indices)
test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator_1,
    output_types=(tf.float32, tf.float32))
test_dataset = test_dataset.map(convert_to_dictionaries)
test_dataset = test_dataset.take(NUMBER_OF_BATCHES)  # TODO

"""BASE MODEL"""
base_model = InceptionResNetV2(weights=None, classes=num_of_classes)
base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""NSL MODEL"""
adv_config = nsl.configs.make_adv_reg_config(multiplier=multiplier,
                                             adv_step_size=adv_step_size,
                                             adv_grad_norm=adv_grad_norm)
adv_model = nsl.keras.AdversarialRegularization(keras.models.clone_model(base_model),  # TODO copied here pay attention
                                                label_keys=[LABEL_INPUT_NAME],
                                                adv_config=adv_config)
adv_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# eval on 1 batch just to 'wakeup' the adversarial model (in order to load weights)
adv_model.evaluate(test_dataset.take(1), verbose=0)  # TODO

"""LOAD DATASET AGAIN BECAUSE OF EVALUATE"""
pre_process = preprocess_input_inception_resnet_v2
test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
test_generator_2 = test_data_gen.flow_from_dataframe(dataframe=test_df, x_col="path", y_col="breed",
                                                     class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                     batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
print(test_generator_2.class_indices)
test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator_2,
    output_types=(tf.float32, tf.float32))
test_dataset = test_dataset.map(convert_to_dictionaries)
test_dataset = test_dataset.take(NUMBER_OF_BATCHES)  # TODO

"""LOAD WEIGHTS"""
base_model.load_weights(r'weights\flat_weights_inception_resnet_v2_80.39%.hdf5')
adv_model.load_weights(r'weights\nsl_weights_inception_resnet_v2_0.2_0.2_l2_79.37%.hdf5')

"""EVAL ON ADVERSARIAL EXAMPLES"""
models_to_eval = {
    'base': base_model,
    'adv-regularized': adv_model.base_model
}

metrics = {
    name: tf.keras.metrics.CategoricalAccuracy()
    for name in models_to_eval.keys()
}

reference_model = nsl.keras.AdversarialRegularization(
    base_model,
    label_keys=[LABEL_INPUT_NAME],
    adv_config=adv_config)

reference_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc'])

"""EVALUATE MODELS CHECK"""  # TODO del section
# TODO use predict instead
# for model_name, model_func in models_to_eval.items():
#     eval_test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
#     eval_test_generator = eval_test_data_gen.flow_from_dataframe(dataframe=test_df, x_col="path", y_col="breed",
#                                                                  class_mode="categorical",
#                                                                  target_size=INPUT_SHAPE[:2],
#                                                                  batch_size=1, shuffle=False)
#     print(eval_test_generator.class_indices)
#
#     eval_test_dataset = tf.data.Dataset.from_generator(
#         lambda: eval_test_generator,
#         output_types=(tf.float32, tf.float32))
#     # eval_test_dataset = eval_test_dataset.map(convert_to_dictionaries)
#     eval_test_dataset = eval_test_dataset.take(len(test_df))
#
#     classes = eval_test_generator.class_indices
#     inverted_classes = dict(map(reversed, classes.items()))
#     predictions = model_func.predict(test_dataset)
#     predictions = tf.argmax(predictions, axis=-1).numpy()
#     inverted_class_predictions = [inverted_classes[i] for i in predictions]
#
#     test_df[model_name] = inverted_class_predictions
#     accuracy = len(test_df[test_df['breed'] == test_df[model_name]]) / len(test_df)
#
#     print(f'{model_name} accuracy: {accuracy}')
#     del eval_test_dataset
#
# exit()

perturbed_images, labels, predictions, probabilities = [], [], [], []

for k, batch in enumerate(test_dataset):
    if k >= NUMBER_OF_BATCHES:
        break
    perturbed_batch = reference_model.perturb_on_batch(batch)  # TODO
    # perturbed_batch = batch  # TODO
    # Clipping makes perturbed examples have the same range as regular ones.
    perturbed_batch[IMAGE_INPUT_NAME] = tf.clip_by_value(
        perturbed_batch[IMAGE_INPUT_NAME], -1.0, 1.0)
    y_true = perturbed_batch.pop(LABEL_INPUT_NAME)
    perturbed_images.append(perturbed_batch[IMAGE_INPUT_NAME].numpy())
    labels.append(y_true.numpy())
    predictions.append({})
    probabilities.append({})
    for name, model in models_to_eval.items():
        y_pred = model(perturbed_batch)
        metrics[name](y_true, y_pred)
        probabilities[-1][name] = y_pred
        predictions[-1][name] = tf.argmax(y_pred, axis=-1).numpy()

for name, metric in metrics.items():
    print('%s model accuracy: %f' % (name, metric.result().numpy()))

# for original images
original_test_data_gen = ImageDataGenerator()
original_test_generator = original_test_data_gen.flow_from_dataframe(dataframe=test_df, x_col="path", y_col="breed",
                                                                     class_mode="categorical",
                                                                     target_size=INPUT_SHAPE[:2],
                                                                     batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
print(original_test_generator.class_indices)

original_test_dataset = tf.data.Dataset.from_generator(
    lambda: original_test_generator,
    output_types=(tf.float32, tf.float32))
original_test_dataset = original_test_dataset.map(convert_to_dictionaries)
images = list(original_test_dataset.take(NUMBER_OF_BATCHES))

for batch_index in range(NUMBER_OF_BATCHES):
    batch_image = perturbed_images[batch_index]
    batch_label = labels[batch_index]
    batch_pred = predictions[batch_index]
    batch_images = images[batch_index]

    batch_size = BATCH_SIZE
    n_col = 4
    n_row = (batch_size + n_col - 1) / n_col

    print('accuracy in batch %d:' % batch_index)
    for name, pred in batch_pred.items():
        print('%s model: %d / %d' % (name, np.sum(tf.argmax(batch_label, axis=-1).numpy() == pred), batch_size))

    # classes = test_generator.class_indices
    # inverted_classes = dict(map(reversed, classes.items()))
    # print(inverted_classes)

    # original code
    plt.figure(figsize=(15, 15))
    for i, (image, y) in enumerate(zip(batch_image, batch_label)):
        y_base = batch_pred['base'][i]
        y_adv = batch_pred['adv-regularized'][i]
        plt.subplot(n_row, n_col, i + 1)
        plt.title('true: %d, base: %d, adv: %d' % (tf.argmax(y).numpy(), y_base, y_adv))
        # TODO the opposite of the pre processed
        plt.imshow(tf.keras.preprocessing.image.array_to_img((image + 1.0) * 127.5))
        plt.axis('off')
    plt.show()

    # regular images
    plt.figure(figsize=(BATCH_SIZE - 1, BATCH_SIZE - 1))
    for i, (image, y) in enumerate(zip(batch_image, batch_label)):
        y_base = batch_pred['base'][i]
        y_adv = batch_pred['adv-regularized'][i]
        plt.subplot(n_row, n_col, i + 1)
        plt.title('true: %d, base: %d, adv: %d' % (tf.argmax(y).numpy(), y_base, y_adv))
        plt.imshow(tf.keras.preprocessing.image.array_to_img(batch_images['input_1'][i]))  # TODO
        plt.axis('off')
    plt.show()
