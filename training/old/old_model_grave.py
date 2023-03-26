# Old vgg16 model --> Tends to overfit on the training data, used a smaller pretrained model and refined some layers
# vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# for layer in vgg16.layers:
# layer.trainable = False

# for layer in vgg16.layers[-4:]:
# layer.trainable = True

model = Sequential([
    data_augmentation,
    vgg16,
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Flatten(),
    layers.Dense(128, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.Dense(num_classes, activation='softmax', name="outputs")
]) if not load_model else keras.models.load_model(load_path)