import tensorflow as tf
import pytest
import os
import logging
from typing import Tuple

from utilities.tools import (load_dataset, create_augmentation_layer, get_data_path_addon, suppress_tf_warnings, resize_dataset)


def test_load_dataset(tmp_path):
    # Create a temporary dataset with 2 classes and 4 images each
    d = tmp_path / "data"
    d.mkdir()
    d1 = d / "class1"
    d1.mkdir()
    d2 = d / "class2"
    d2.mkdir()
    for i in range(4):
        p1 = d1 / f"img{i + 1}.jpg"
        p1.write_text("fake image data")
        p2 = d2 / f"img{i + 1}.jpg"
        p2.write_text("fake image data")

    train_ds, val_ds, class_names = load_dataset(str(d), 2, 32, 32)

    assert len(train_ds) == 4
    assert len(val_ds) == 1
    assert set(class_names) == {"class1", "class2"}


def test_create_augmentation_layer():
    data_augmentation = create_augmentation_layer(32, 32)
    assert len(data_augmentation.layers) == 3
    assert isinstance(data_augmentation.layers[0], tf.keras.layers.RandomFlip)
    assert isinstance(data_augmentation.layers[1], tf.keras.layers.RandomRotation)
    assert isinstance(data_augmentation.layers[2], tf.keras.layers.RandomZoom)


def test_get_data_path_addon():
    assert get_data_path_addon("car_type") == "Porsche"
    assert get_data_path_addon("all_specific_model_variants") == "Porsche_more_classes"
    assert get_data_path_addon("specific_model_variants") == "Porsche_variants"
    assert get_data_path_addon("pre_filter") == "pre_filter"

    with pytest.raises(ValueError):
        get_data_path_addon("invalid_name")


def test_suppress_tf_warnings():
    suppress_tf_warnings()
    assert logging.getLogger('tensorflow').isEnabledFor(logging.ERROR)
    assert os.environ['TF_CPP_MIN_LOG_LEVEL'] == '3'
    assert os.environ['KMP_AFFINITY'] == 'noverbose'


def test_resize_dataset():
    def create_dummy_data() -> Tuple[tf.data.Dataset, tf.Tensor, tf.Tensor]:
        num_samples = 10
        img_height, img_width = 28, 28
        images = tf.random.uniform((num_samples, img_height, img_width, 3), minval=0, maxval=255, dtype=tf.float32)
        labels = tf.random.uniform((num_samples,), minval=0, maxval=10, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        return dataset, images, labels

    dataset, images, labels = create_dummy_data()
    new_height, new_width = 64, 64
    resized_dataset = resize_dataset(dataset, new_height, new_width)

    for i, (x, y) in enumerate(resized_dataset):
        assert x.shape == (new_height, new_width, 3)
        assert y == labels[i]
