from tensorflow import keras
from tensorflow.keras.applications.xception import Xception


def get_basic_model(height: int, width: int, num_classes: int, metrics=None, biases='zeros') -> keras.Model:
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(height, width, 3),
        pooling=None,
        classes=num_classes)
    flat = keras.layers.Flatten()(base_model.output)
    classifier_module = keras.layers.Dense(256, activation='relu')(flat)
    classifier_module = keras.layers.Dropout(.3)(classifier_module)
    classifier_module = keras.layers.Dense(
        num_classes,
        activation='softmax',
        bias_initializer=biases)(classifier_module)
    model = keras.Model(base_model.input, outputs=classifier_module)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'] if metrics is None else metrics)

    return model
