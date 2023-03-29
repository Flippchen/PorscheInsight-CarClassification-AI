from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_data_generator(train_dir: str, img_height: int, img_width: int, batch_size: int):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=(img_height, img_width),
                                                        batch_size=batch_size,
                                                        class_mode='c')
    return train_generator


def val_data_generator(val_dir: str, img_height: int, img_width: int, batch_size: int):
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_directory(val_dir,
                                                                  target_size=(img_height, img_width),
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical')
    return validation_generator

# history = model.fit(train_generator,
#                    steps_per_epoch=len(train_generator),
#                   epochs=50,
#                   validation_data=validation_generator,
#                   validation_steps=len(validation_generator))
