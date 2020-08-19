import pandas as pd
import os


def voc_cats_dogs_images():
    """
    get the images of cats and dogs from PASCAL VOC
    filter images with both cat and dog

    returns df
    path      | cat/dog  | breed  | dataset
    ___________________________________
    file path | dog     | 3      | voc
    file path | cat     | 6      | voc
    :return: data frame with voc images paths
    """
    cat_path = os.path.join('VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'cat_trainval.txt')
    dog_path = os.path.join('VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'dog_trainval.txt')

    # a set for the images names for cats and dogs
    voc_cats_images_names = set()
    voc_dogs_images_names = set()

    for animal, path in {'cat': cat_path, 'dog': dog_path}.items():
        with open(path) as file:
            for line in file.readlines():
                splited = line.split()
                file_name, has_animal = splited[0] + '.jpg', splited[1]
                if has_animal == '1':
                    if animal == 'cat':
                        voc_cats_images_names.add(os.path.join('VOCdevkit', 'VOC2012', 'JPEGImages', file_name))
                    else:
                        voc_dogs_images_names.add(os.path.join('VOCdevkit', 'VOC2012', 'JPEGImages', file_name))

    # filter pictures containing both cats and dogs together
    intersection = voc_cats_images_names.intersection(voc_dogs_images_names)

    voc_cats_images_names -= intersection
    voc_dogs_images_names -= intersection

    df = pd.DataFrame(columns=['path', 'cat/dog', 'breed', 'dataset'])
    for cat in voc_cats_images_names:
        df = df.append({'path': cat, 'cat/dog': 'cat', 'breed': None, 'dataset': 'voc'}, ignore_index=True)

    for dog in voc_dogs_images_names:
        df = df.append({'path': dog, 'cat/dog': 'dog', 'breed': None, 'dataset': 'voc'}, ignore_index=True)

    return df


def oxford_cats_dogs_images():
    """
    path      | cat/dog  | breed  | dataset
    ___________________________________
    file path | dog     | 3      | oxford
    file path | cat     | 6      | oxford
    :return: data frame with oxford images paths
    """
    df = pd.DataFrame(columns=['path', 'cat/dog', 'breed', 'dataset'])
    path = os.path.join('annotations', 'list.txt')

    with open(path) as file:
        for line in file.readlines():
            if line[0] == '#':
                continue
            splited = line.split()
            file_name, breed, cat_or_dog = splited[0] + '.jpg', splited[1], splited[2]
            file_path = os.path.join('images', file_name)
            df = df.append({'path': file_path, 'cat/dog': 'cat' if cat_or_dog == '1' else 'dog',
                            'breed': breed, 'dataset': 'oxford'}, ignore_index=True)

    return df


if __name__ == "__main__":
    df_voc = voc_cats_dogs_images()
    df_oxford = oxford_cats_dogs_images()
    df = pd.concat([df_voc, df_oxford])
