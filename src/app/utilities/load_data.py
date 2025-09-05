import os
import random

import cv2
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def crop_image(image):
    h, w, _ = image.shape
    crop_factor = random.uniform(
        0.8, 1
    )  # generate a random crop factor between 0.8 and 1
    cropped_image = cv2.resize(
        image, (int(w), int(h * crop_factor))
    )  # crop image by random factor
    resized_image = cv2.resize(
        cropped_image, (w, h)
    )  # resize cropped image back to original size
    return resized_image


def __gen_generator(cfg, type_set, train_dtgen, df):
    """Generate a generator to train the model
    :param cfg: Config class
    :param type_set: type of data to be generated (training or validation)
    :param train_dtgen: ImageDataGenerator object
    :param df: dataframe with the files and the labels
    :return: generator
    """

    gen = train_dtgen.flow_from_dataframe(
        df,
        x_col="filename",
        y_col="label",
        target_size=(cfg.target_size, cfg.target_size),
        batch_size=cfg.batch_size,
        class_mode="categorical",
        interpolation="nearest",
        subset=type_set,
    )  # set as training data
    return gen


def __gen_df(label, path):
    """Generate a dataframe with the files and the label
    :param label: label of the files
    :param path: path to the files
    :return: dataframe with the files and the label
    """

    file_and_label = []
    for filename in os.listdir(path):
        file_and_label.append((path + "/" + filename, label))
    df = pd.DataFrame(file_and_label, columns=["filename", "label"])
    return df


def gen_data(cfg, pato, EVAL=False):
    """Generate data to train the model
    :param cfg: Config class
    :param pato: pathology to be trained
    :param EVAL: if True, the data is generated for evaluation (smaller data)
    :return: train_gen and val_gen
    """

    split = 0.2 if not EVAL else 0.1
    train_dtgen = ImageDataGenerator(
        fill_mode="nearest", validation_split=split
    )  # set validation split

    # check the pathology to be trained and then load files with DF to train
    df_pato = __gen_df(pato, cfg.source + pato)
    df_norm = __gen_df("normais", cfg.source + "normais")
    df = pd.concat([df_pato, df_norm], ignore_index=True)

    # keep 10k images for each class on df randomly
    # df = df.groupby('label').apply(lambda x: x.sample(n=17000, random_state=1)).reset_index(drop=True)

    train_gen = __gen_generator(cfg, "training", train_dtgen, df)
    val_gen = __gen_generator(cfg, "validation", train_dtgen, df)

    return train_gen, val_gen


def gen_data_kfold(cfg, df, train_idx, val_idx):
    """Generate data for a specific fold in K-fold cross-validation.
    :param cfg: Config class
    :param df: dataframe with the files and the labels
    :param train_idx: indices for the training data
    :param val_idx: indices for the validation data
    :return: train_gen and val_gen
    """
    train_dtgen = ImageDataGenerator(fill_mode="nearest")
    val_dtgen = ImageDataGenerator(fill_mode="nearest")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_gen = __gen_generator(cfg, None, train_dtgen, train_df)
    val_gen = __gen_generator(cfg, None, val_dtgen, val_df)

    return train_gen, val_gen
