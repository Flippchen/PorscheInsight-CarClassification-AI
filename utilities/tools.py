import pathlib
import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras import layers
import os
import logging


def load_dataset(path: str, batch_size: int, img_height: int, img_width: int) -> tuple[tf.data.Dataset, tf.data.Dataset, list]:
    """
    :param path: Path to the Dataset folder
    :param batch_size: Integer which defines how many Images are in one Batch
    :param img_height: Height of the images to be loaded with
    :param img_width: Width of the images to be loaded with
    :return: Tuple of train, val Dataset and Class names
    """
    data_dir = pathlib.Path(path)
    # if "more_classes" in path:
    #    image_count = len(list(data_dir.glob('*/*/*.jpg')))
    # else:
    #    image_count = len(list(data_dir.glob('*/*/*/*.jpg')))

    # print("Image count:", image_count)
    # cars = list(data_dir.glob('*/*/*/*.jpg'))
    # PIL.Image.open(str(cars[0]))
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    return train_ds, val_ds, class_names


def load_explainer_data(path: str, batch_size: int, img_height: int, img_width: int, shuffle: int = 10000, number_images: int = 1000) -> list[np.ndarray]:
    """
    :param path: Path to the dataset folder
    :param batch_size: Defines how many images are in one batch
    :param img_height: Height of the images to be loaded with
    :param img_width: Width of the images to be loaded with
    :param shuffle: How many times the dataset get shuffled before return
    :param number_images: How many images to return
    :return: List of numpy representations of images
    """
    data_dir = pathlib.Path(path)
    if "more_classes" in path:
        image_count = len(list(data_dir.glob('*/*/*.jpg')))
    else:
        image_count = len(list(data_dir.glob('*/*/*/*.jpg')))

    print("Image count:", image_count)

    data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Calculate the number of images in the dataset after applying the batch_size
    num_images_in_dataset = len(data) * batch_size
    # Create warning if take is greater than the number of images in the dataset
    if number_images > num_images_in_dataset:
        warnings.warn(f"{number_images} is greater than the number of images in the dataset. It will be set to maximum number of images in the dataset.")
    # Calculate the number of batches to take based on the take value
    take = (number_images + batch_size - 1) // batch_size
    data = data.shuffle(shuffle).take(take)
    images = []
    for image_batch, labels_batch in data:
        images.append(image_batch)

    return images


def load_image_subset(path: str, batch_size: int, img_height: int, img_width: int, shuffle: int = 10000, number_images: int = 1000) -> tf.data.Dataset:
    """
    :param path: Path to dataset folder
    :param batch_size: How many images are in one batch
    :param img_height: Height of the images to be loaded with
    :param img_width: Width of the images to be loaded with
    :param shuffle: How many times the dataset gets shuffled before return
    :param number_images: Number of images to be returned
    :return: Subset of Dataset
    """
    data_dir = pathlib.Path(path)
    # if "more_classes" in path:
    #    image_count = len(list(data_dir.glob('*/*/*.jpg')))
    # else:
    #    image_count = len(list(data_dir.glob('*/*/*/*.jpg')))

    # print("Image count:", image_count)

    data = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # Calculate the number of images in the dataset after applying the batch_size
    num_images_in_dataset = len(data) * batch_size
    # Create warning if take is greater than the number of images in the dataset
    if number_images > num_images_in_dataset:
        warnings.warn(f"{number_images} is greater than the number of images in the dataset. It will be set to maximum number of images in the dataset.")

    # Calculate the number of batches to take based on the take value
    take = (number_images + batch_size - 1) // batch_size

    data = data.shuffle(shuffle).take(take)

    return data


def show_sample_batch(train_ds: tf.data.Dataset, class_names: list) -> None:
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()


def show_batch_shape(train_ds: tf.data.Dataset) -> None:
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break


def create_augmentation_layer(img_height: int, img_width: int) -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )


def show_augmented_batch(train_ds, data_augmentation) -> None:
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
        plt.show()


def plot_model_score(history, name: str, model_type: str) -> None:
    if model_type == "all_specific_model_variants":
        plot_save_path = f'../models/all_model_variants/results/acc-loss-{name}-model.png'
    elif model_type == "model_type":
        plot_save_path = f'../models/car_types/results/acc-loss-{name}-model.png'
    elif model_type == "specific_model_variants":
        plot_save_path = f'../models/model_variants/results/acc-loss-{name}-model.png'
    else:
        raise ValueError("model_type must be one of 'all_specific_model_variants', 'model_type' or 'specific_model_variants'")
    # Read history and plot model score
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(plot_save_path)


def suppress_tf_warnings():
    # Suppress TensorFlow INFO and WARNING logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["KMP_AFFINITY"] = "noverbose"

    # Suppress Python logging warnings
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    # Suppress any deprecated function warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, model_type: str, name: str) -> None:
    if model_type == "all_specific_model_variants":
        title = f"Confusion Matrix for All Specific Model Variants"
        plot_save_path = f'cm_all_specific_model_variants-{name}.png'
        fig_size = (25, 25)
        sns.set(font_scale=0.7)
    elif model_type == "model_type":
        title = f"Confusion Matrix for Car Type"
        plot_save_path = f'cm_car_type-{name}.png'
        fig_size = (10, 10)
        sns.set(font_scale=1.0)
    elif model_type == "specific_model_variants":
        title = f"Confusion Matrix for Specific Model Variants"
        plot_save_path = f'cm_specific_model_variants-{name}.png'
        fig_size = (10, 10)
        sns.set(font_scale=0.7)
    else:
        raise ValueError("Invalid model type")

    # Convert the confusion matrix from an array to a list
    cm_list = cm.tolist()

    # Normalize the confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix
    plt.figure(figsize=fig_size)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig(f"results/{plot_save_path}")


def resize_dataset(data: tf.data.Dataset, img_height: int, img_width: int) -> tf.data.Dataset:
    data = data.map(lambda x, y: (tf.image.resize(x, (img_height, img_width)), y))
    return data


def get_data_path_addon(name: str) -> str:
    if name == "car_type":
        return "Porsche"
    elif name == "all_specific_model_variants":
        return "Porsche_more_classes"
    elif name == "specific_model_variants":
        return "Porsche_variants"
    else:
        raise ValueError("Invalid model name")
