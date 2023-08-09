from tensorflow import keras
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

from typing import Callable


def get_basic_model(height: int, width: int, num_classes: int) -> keras.Model:
    base_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(height, width, 3))
    flat = keras.layers.Flatten()(base_model.output)
    locator_module = keras.layers.Dense(2048, activation='relu')(flat)
    locator_module = keras.layers.Dropout(.3)(locator_module)
    locator_module = keras.layers.Dense(num_classes, activation='softmax')(locator_module)
    model = keras.Model(base_model.input, outputs=locator_module)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def get_model_partly_frozen(height: int, width: int, num_classes: int) -> keras.Model:
    base_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(height, width, 3))

    for cnt in range(len(base_model.layers) // 2, len(base_model.layers)):
        base_model.layers[cnt].trainable = False

    flat = keras.layers.Flatten()(base_model.output)
    locator_module = keras.layers.Dense(4608, activation='relu')(flat)
    locator_module = keras.layers.Dropout(.3)(locator_module)
    locator_module = keras.layers.Dense(2048, activation='relu')(locator_module)
    locator_module = keras.layers.Dropout(.3)(locator_module)
    locator_module = keras.layers.Dense(num_classes, activation='softmax')(locator_module)
    model = keras.Model(base_model.input, outputs=locator_module)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def get_model_with_attention(height: int, width: int, num_classes: int, freezer: Callable = None) -> keras.Model:
    def get_attention_module(prev: keras.layers.Layer) -> keras.layers.Layer:
        gap_layer = keras.layers.GlobalAveragePooling2D()(prev)
        gap_layer_res = keras.layers.Reshape((1, 1, 1536))(gap_layer)
        dense = keras.layers.Dense(1536, activation='relu')(gap_layer_res)
        dense = keras.layers.Dense(1536, activation='softmax')(dense)
        mul_layer = keras.layers.Multiply()([prev, dense])

        return mul_layer

    base_model = InceptionResNetV2(
        include_top=False,
        weights='imagenet',
        input_shape=(height, width, 3))

    if freezer is not None:
        freezer(base_model)

    attention_module = get_attention_module(base_model.output)
    flat = keras.layers.Flatten()(attention_module)
    locator_module = keras.layers.Dense(2048, activation='relu')(flat)
    locator_module = keras.layers.Dropout(.3)(locator_module)
    locator_module = keras.layers.Dense(num_classes, activation='softmax')(locator_module)
    model = keras.Model(base_model.input, outputs=locator_module)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model
