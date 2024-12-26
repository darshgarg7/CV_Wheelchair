import logging
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras_tuner import Hyperband
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from models.data_utils import augment_data_with_gan
from models.model_utils import build_model, monitor_model
from devcontainer.deployment_utils import save_model_for_serving

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def hyperparameter_tuning(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    logger.info("Initializing hyperparameter tuning...")
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
    logger.info(f"Best hyperparameters: {best_hyperparams.values}")
    return best_hyperparams

def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray):
    loss, acc = model.evaluate(X_val, y_val, verbose=1)
    logger.info(f"Validation Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred_classes, average='macro')
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

def train_model():
    try:
        logger.info("Generating synthetic data...")
        num_samples = 1000
        images = np.random.rand(num_samples, 224, 224, 3) * 255
        gestures = np.random.randint(0, 4, size=num_samples)

        logger.info("Augmenting data using GAN...")
        images_augmented, gestures_augmented = augment_data_with_gan(images, gestures, real_to_synthetic_ratio=0.5)

        if images_augmented.shape[0] != gestures_augmented.shape[0]:
            raise ValueError("Data mismatch between images and labels.")

        logger.info("Shuffling and splitting data into training and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            images_augmented, gestures_augmented, test_size=0.2, random_state=42
        )

        best_hyperparams = hyperparameter_tuning(X_train, y_train, X_val, y_val)
        
        logger.info("Building model with optimized parameters...")
        model = build_model(hyperparams=best_hyperparams)

        logger.info("Configuring callbacks...")
        callbacks = [
            ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', verbose=1),
            EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True, verbose=1),
            TensorBoard(log_dir='./logs', update_freq='epoch')
        ]

        logger.info("Starting model training...")
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, callbacks=callbacks)

        logger.info("Training complete. Logging performance metrics...")
        for epoch in range(len(history.history['loss'])):
            logger.info(f"Epoch {epoch+1}: Loss = {history.history['loss'][epoch]}, Accuracy = {history.history['accuracy'][epoch]}")

        evaluate_model(model, X_val, y_val)
        
        logger.info("Saving the trained model for deployment...")
        save_model_for_serving(model, 'gesture_recognition_model')

        logger.info("Monitoring model on validation data...")
        monitor_model(model, X_val, y_val)

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)

if __name__ == "__main__":
    train_model()
