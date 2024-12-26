import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet, EfficientNetLite

def build_model(hyperparams: dict = None) -> models.Model:
    """
    Build a gesture recognition model using MobileNet or EfficientNet-Lite as the backbone.
    The choice of base model is fixed to MobileNet, but you can easily swap it with EfficientNet-Lite by changing the model initialization.

    :param hyperparams: Dictionary of hyperparameters to customize model behavior (e.g., learning rate, dropout, dense units).
    :return: A compiled Keras model instance.
    """
    
    # Default hyperparameters if none are provided
    if hyperparams is None:
        hyperparams = {
            'trainable': False,
            'dropout': 0.5,
            'dense_units': 512,
            'lr': 1e-4
        }

    # Choose base model: MobileNet or EfficientNet-Lite
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = hyperparams.get('trainable', False)

    # Build model architecture
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Reduce dimensionality from feature maps
        layers.Dropout(hyperparams.get('dropout', 0.5)),  # Regularization to prevent overfitting
        layers.Dense(hyperparams.get('dense_units', 512), activation='swish'),  # Fully connected layer
        layers.Dropout(hyperparams.get('dropout', 0.5)),  # Another dropout layer
        layers.Dense(3, activation='softmax')  # Output layer with 3 units for 3 gesture classes
    ])
    
    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    model.compile(
        optimizer=Adam(learning_rate=hyperparams.get('lr', 1e-4)),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
