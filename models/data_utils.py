from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, Embedding, Concatenate
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import Model

def augment_data_with_gan(images: np.ndarray, gestures: np.ndarray, num_samples: int = 1000, real_to_synthetic_ratio: float = 0.5) -> tuple:
    """
    Generate synthetic images using a GAN and augment the dataset.
    
    :param images: Original dataset images (n_samples, height, width, channels).
    :param gestures: Corresponding labels (n_samples,).
    :param num_samples: Number of synthetic images to add (default: 1000).
    :param real_to_synthetic_ratio: Ratio of real to synthetic images.
    :return: Tuple (augmented_images, augmented_labels).
    """
    gan_generator = build_conditional_gan_generator()  # Use conditional GAN
    noise = np.random.randn(num_samples, 100)  # Random latent vectors
    labels = np.random.choice(np.unique(gestures), size=num_samples)  # Random gesture labels
    generated_images = gan_generator.predict([noise, labels])

    # Rescale generated images from [-1, 1] to [0, 255]
    generated_images = np.clip((generated_images + 1) * 127.5, 0, 255)

    if images.shape[1:] != generated_images.shape[1:]:
        raise ValueError("Original and generated image dimensions do not match.")
    
    # Shuffle real and synthetic samples
    num_real_samples = int((1 - real_to_synthetic_ratio) * len(images))
    num_synthetic_samples = len(images) - num_real_samples

    augmented_images = np.vstack([images[:num_real_samples], generated_images[:num_synthetic_samples]])
    augmented_labels = np.hstack([gestures[:num_real_samples], labels[:num_synthetic_samples]])
    
    return augmented_images, augmented_labels

def build_conditional_gan_generator() -> Model:
    """
    Build a conditional GAN generator model.
    
    :return: A Keras model that generates images from latent space noise and gesture labels.
    """
    noise_input = layers.Input(shape=(100,))
    label_input = layers.Input(shape=(1,))
    label_embedding = Embedding(3, 100)(label_input)  # Assuming 3 classes (gestures)
    merged_input = Concatenate()([noise_input, label_embedding])
    
    x = Dense(128, activation='relu')(merged_input)
    x = Dense(7 * 7 * 256, activation='relu')(x)
    x = Reshape((7, 7, 256))(x)
    x = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='relu')(x)
    generated_image = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='tanh')(x)

    model = Model(inputs=[noise_input, label_input], outputs=generated_image)
    return model
