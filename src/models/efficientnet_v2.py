from tensorflow import keras
from tensorflow.keras.applications.efficientnet_v2 import \
    EfficientNetV2M, \
    EfficientNetV2S, \
    EfficientNetV2B3


def _get_model(num_classes: int, base_model: keras.Model) -> keras.Model:
    flat = keras.layers.Flatten()(base_model.output)
    classifier_module = keras.layers.Dense(2048, activation='relu')(flat)
    classifier_module = keras.layers.Dropout(.3)(classifier_module)
    classifier_module = keras.layers.Dense(num_classes, activation='softmax')(classifier_module)
    model = keras.Model(base_model.input, outputs=classifier_module)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def get_v2m_model(height: int, width: int, num_classes: int) -> keras.Model:
    base_model = EfficientNetV2M(
        include_top=False,
        weights='imagenet',
        input_shape=(height, width, 3))

    return _get_model(num_classes, base_model)


def get_v2s_model(height: int, width: int, num_classes: int) -> keras.Model:
    base_model = EfficientNetV2S(
        include_top=False,
        weights='imagenet',
        input_shape=(height, width, 3))

    return _get_model(num_classes, base_model)


def get_v2b3_model(height: int, width: int, num_classes: int) -> keras.Model:
    base_model = EfficientNetV2B3(
        include_top=False,
        weights='imagenet',
        input_shape=(height, width, 3))

    return _get_model(num_classes, base_model)
