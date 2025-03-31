import logging
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras_tuner import Hyperband
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from models.data_utils import augment_data_with_gan, ConditionalGAN
from models.model_utils import build_model, monitor_model
from devcontainer.deployment_utils import save_model_for_serving
import tensorflow as tf

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """
    Perform hyperparameter tuning using KerasTuner's Hyperband method.
    """
    try:
        logger.info("Initializing hyperparameter tuning using Hyperband...")
        tuner = Hyperband(
            build_model,
            objective='val_accuracy',
            max_epochs=10,
            factor=3,
            directory='hyperparam_tuning',
            project_name='gesture_recognition'
        )
        tuner.search(X_train, y_train, validation_data=(X_val, y_val), verbose=1)
        best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info(f"Best hyperparameters found: {best_hyperparams.values}")
        return best_hyperparams
    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}", exc_info=True)
        raise

def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray):
    """
    Evaluates the model's performance on validation data.
    """
    try:
        logger.info("Evaluating model on validation data...")
        loss, acc = model.evaluate(X_val, y_val, verbose=1)
        logger.info(f"Validation Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        y_pred = model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred_classes, average='macro')
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        raise

def train_model():
    """
    Main function to orchestrate the model training pipeline.
    """
    try:
        logger.info("Loading and augmenting real gesture data...")
        images, gestures = augment_data_with_gan(num_samples=1000)
        logger.info("Initializing Conditional GAN for additional augmentation...")
        conditional_gan = ConditionalGAN(input_shape=(224, 224, 3), num_classes=4)
        conditional_gan.train(images, gestures, epochs=10, batch_size=32)
        logger.info("Generating synthetic data with Conditional GAN...")
        synthetic_images, synthetic_labels = conditional_gan.generate_data(num_samples=500)

        # Combine real and synthetic data
        images_augmented = np.concatenate([images, synthetic_images], axis=0)
        gestures_augmented = np.concatenate([gestures, synthetic_labels], axis=0)

        # Data consistency check
        assert images_augmented.shape[0] == gestures_augmented.shape[0], "Mismatch between the number of images and labels."

        logger.info("Shuffling and splitting data into training and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            images_augmented, gestures_augmented, test_size=0.2, random_state=42
        )

        # Perform hyperparameter tuning
        best_hyperparams = hyperparameter_tuning(X_train, y_train, X_val, y_val)

        # Build and train the model with optimized hyperparameters
        logger.info("Building model with optimized hyperparameters...")
        model = build_model(hyperparams=best_hyperparams)

        logger.info("Configuring callbacks for model training...")
        callbacks = [
            ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', verbose=1),
            EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True, verbose=1),
            TensorBoard(log_dir='./logs', update_freq='epoch')
        ]
        callbacks.extend(monitor_model(monitor='val_accuracy'))

        logger.info("Training the model...")
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=callbacks)

        logger.info("Training completed. Logging metrics...")
        for epoch in range(len(history.history['loss'])):
            logger.info(f"Epoch {epoch+1}: Loss = {history.history['loss'][epoch]}, Accuracy = {history.history['accuracy'][epoch]}")

        # Evaluate the model
        evaluate_model(model, X_val, y_val)

        # Save the model for deployment
        logger.info("Saving the model for deployment...")
        save_model_for_serving(model, 'models/gesture_recognition_model')

    except AssertionError as e:
        logger.error(f"Data inconsistency: {e}", exc_info=True)
    except tf.errors.InvalidArgumentError as e:
        logger.error(f"TensorFlow error: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)

if __name__ == "__main__":
    train_model()
