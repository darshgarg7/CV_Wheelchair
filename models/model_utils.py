from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet, EfficientNetLite
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from models.gesture_model import train_model

logger = logging.getLogger()

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

def monitor_model(monitor: str = 'val_loss', patience: int = 5, restore_best_weights: bool = True) -> list:
    """
    Creates and returns the callbacks for monitoring the model during training.

    :param monitor: The metric to monitor during training, either 'val_loss', 'val_accuracy', etc.
    :param patience: Number of epochs with no improvement after which training will be stopped.
    :param restore_best_weights: Whether to restore model weights from the epoch with the best value of the monitored metric.
    :return: List of Keras callbacks for monitoring the model.
    """
    logger.info(f"Creating callbacks to monitor {monitor}...")
    
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor=monitor, verbose=1, mode='max' if 'accuracy' in monitor else 'min')
    early_stopping = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=restore_best_weights, verbose=1, mode='max' if 'accuracy' in monitor else 'min')
    tensorboard = TensorBoard(log_dir='./logs', update_freq='epoch')

    logger.info(f"Callbacks configured for monitoring {monitor}.")
    
    # Return the list of callbacks
    return [checkpoint, early_stopping, tensorboard]

def plot_grad_cam(model, image, conv_dw_13_relu, label_index=None):
    """
    Plots the Grad-CAM heatmap for a specific model prediction.

    :param model: The trained Keras model.
    :param image: Input image for which Grad-CAM will be plotted, shape (224, 224, 3).
    :param last_conv_layer_name: Name of the last convolutional layer in the model.
    :param label_index: Index of the class to explain. If None, the predicted class is used.
    """
    # Check if the model contains the last convolutional layer
    if last_conv_layer_name not in [layer.name for layer in model.layers]:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found in the model.")

    # Create a model that returns the last convolutional layer's output and the final predictions
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Add a batch dimension to the image and preprocess
    image_batch = np.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:
        # Watch the image for gradients
        tape.watch(image_batch)
        
        # Get the model's predictions and convolutional outputs
        conv_outputs, predictions = grad_model(image_batch)
        
        if label_index is None:
            label_index = np.argmax(predictions[0])  # Predicted class index if not specified
        
        class_channel = predictions[:, label_index]

    # Get the gradients of the class output w.r.t the convolutional outputs
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Pool the gradients over all axes (height, width)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Get the convolutional layer outputs and compute the weighted sum of gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)  # ReLU activation
    heatmap /= np.max(heatmap)  # Normalize to range 0-1

    # Plot the heatmap
    plt.imshow(image / 255.0)  # Original image normalized for display
    plt.imshow(heatmap, cmap='jet', alpha=0.4)  # Overlay heatmap
    plt.axis('off')
    plt.title(f'Grad-CAM for Class {label_index}')
    plt.show()

if __name__ == "__main__":
    model = train_model()
    image = np.random.rand(224, 224, 3)

    # Call Grad-CAM function
    plot_grad_cam(model, image, last_conv_layer_name='conv_pw_13_relu', label_index=1)  # Adjust layer name and index as needed
