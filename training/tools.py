import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import layers


def load_dataset(path: str, batch_size: int, img_height: int, img_width: int):
    data_dir = pathlib.Path(path)

    image_count = len(list(data_dir.glob('*/*/*/*.jpg')))
    print("Image count:", image_count)

    cars = list(data_dir.glob('*/*/*/*.jpg'))
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


def show_sample_batch(train_ds, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
        plt.show()


def show_batch_shape(train_ds):
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break


def create_augmentation_layer(img_height: int, img_width: int):
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
def show_augmented_batch(train_ds, data_augmentation):
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
        plt.show()
