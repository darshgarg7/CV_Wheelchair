from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet, EfficientNetLite


def build_model(hyperparams: dict = None, backbone: str = 'MobileNet') -> models.Model:
    """
    Build a gesture recognition model with a customizable backbone.

    :param hyperparams: Dictionary with required keys ['trainable', 'dropout', 'dense_units', 'lr'].
    :param backbone: 'MobileNet' or 'EfficientNetLite' for the base model.
    :return: A compiled Keras model.
    """
    
    if hyperparams is None:
        hyperparams = {
            'trainable': False,
            'dropout': 0.5,
            'dense_units': 512,
            'lr': 1e-4
        }

    if backbone == 'MobileNet':
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif backbone == 'EfficientNetLite':
        base_model = EfficientNetLite(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("Unsupported backbone. Choose MobileNet or EfficientNetLite.")

    base_model.trainable = hyperparams['trainable']
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(hyperparams['dropout']),
        layers.Dense(hyperparams['dense_units'], activation='swish'),
        layers.Dropout(hyperparams['dropout']),
        layers.Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=hyperparams['lr']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
