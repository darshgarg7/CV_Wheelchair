from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB7

def build_model(hyperparams=None):
    """
    Build a gesture recognition model using EfficientNetB7 as the backbone.
    Hyperparameters are applied if provided.
    
    :param hyperparams: Dictionary of hyperparameters.
    :return: Keras model instance.
    """
    base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = hyperparams.get('trainable', False)

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(hyperparams.get('dropout', 0.5)),
        layers.Dense(hyperparams.get('dense_units', 512), activation='swish'),
        layers.Dropout(hyperparams.get('dropout', 0.5)),
        layers.Dense(3, activation='softmax')  # For three gesture classes
    ])
    
    model.compile(optimizer=Adam(learning_rate=hyperparams.get('lr', 1e-4)),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def monitor_model(model, X_val, y_val):
    """
    Monitor model performance on the validation set.
    
    :param model: Trained model instance.
    :param X_val: Validation data.
    :param y_val: Validation labels.
    """
    loss, acc = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation Loss: {loss:.4f}, Accuracy: {acc:.4f}")
