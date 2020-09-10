from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

INPUT_SHAPE = [299, 299, 3]

path = r'images/german_shorthaired_126.jpg'
df = pd.DataFrame(columns=['path', 'cat/dog'])
df = df.append({'path': path, 'cat/dog': 'dog'}, ignore_index=True)

generator = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                               width_shift_range=0.2, height_shift_range=0.2,
                               rotation_range=20, brightness_range=[0.7, 1.1])

flow = generator.flow_from_dataframe(dataframe=df, x_col="path", y_col="cat/dog",
                                     class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                     batch_size=1)

original_image = tf.image.resize(plt.imread(path), INPUT_SHAPE[:2]).numpy().astype('int')
image1 = flow.next()[0][0].astype('int')
image2 = flow.next()[0][0].astype('int')
image3 = flow.next()[0][0].astype('int')

plt.figure(figsize=(5, 5))
for i, image in enumerate([original_image, image1, image2, image3]):
    plt.subplot(2, 2, i + 1)
    if i == 0:
        plt.title('original', {'size': 14})
    else:
        plt.title('augmented')
    plt.imshow(image)
    plt.axis('off')
plt.tight_layout()
plt.show()
