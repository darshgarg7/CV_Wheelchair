from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose
from tensorflow.keras import Sequential
import numpy as np

def augment_data_with_gan(images, gestures, num_samples=1000):
    """
    Generate synthetic images using a GAN and augment the dataset.
    
    :param images: Original dataset images (n_samples, height, width, channels).
    :param gestures: Corresponding labels (n_samples,).
    :param num_samples: Number of synthetic images to add (default: 1000).
    :return: Tuple (augmented_images, augmented_labels).
    """
    gan_generator = build_gan_generator()
    noise = np.random.randn(num_samples, 100)  # Random latent vectors
    generated_images = gan_generator.predict(noise)

    # Rescale generated images from [-1, 1] to [0, 255]
    generated_images = np.clip((generated_images + 1) * 127.5, 0, 255)

    # Validate shape consistency
    if images.shape[1:] != generated_images.shape[1:]:
        raise ValueError("Original and generated image dimensions do not match.")
    
    synthetic_labels = np.random.choice(np.unique(gestures), size=num_samples)  # Randomized labels
    augmented_images = np.vstack([images, generated_images])
    augmented_labels = np.hstack([gestures, synthetic_labels])
    
    return augmented_images, augmented_labels

def build_gan_generator():
    """
    Build a simple GAN generator model.
    
    :return: A Keras model that generates images from latent space noise.
    """
    generator = Sequential([
        Dense(128, activation='relu', input_dim=100),
        Dense(7 * 7 * 256, activation='relu'),
        Reshape((7, 7, 256)),
        Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', activation='relu'),
        Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),
        Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh')  # Output layer
    ])
    return generator
