import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_management.database import DatabaseManager

class ConditionalGANGenerator:
    def __init__(self, latent_dim: int = 100, num_classes: int = 4, img_shape: tuple = (28, 28, 3), db_manager=None):
        """
        Initializes the Conditional GAN Generator class.
        """
        if latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.db_manager = db_manager
        self.generator = self._build_generator()

    def _build_generator(self) -> models.Model:
        """
        Builds and returns the GAN generator model conditioned on class labels.
        """
        # Noise input vector
        noise_input = tf.keras.Input(shape=(self.latent_dim,), name="Noise_Input")
        # Class label input
        label_input = tf.keras.Input(shape=(1,), name="Label_Input")
        # Embed label input
        label_embedding = layers.Embedding(input_dim=self.num_classes, output_dim=self.latent_dim)(label_input)
        label_embedding = layers.Reshape((self.latent_dim,))(label_embedding)
        # Merge noise and label inputs
        merged_input = layers.Add()([noise_input, label_embedding])
        # Dense layer
        x = layers.Dense(256, activation='relu')(merged_input)
        x = layers.Reshape((16, 16, 1))(x)  # Reshaping to a 2D feature map
        # Transposed convolution layers to upsample
        x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        # Final output layer with tanh activation to output image
        img_output = layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='tanh')(x)
        # Create model and return
        model = models.Model(inputs=[noise_input, label_input], outputs=img_output, name="Conditional_Generator")
        model.summary()  # Print the model summary for better transparency
        return model

    def generate_synthetic_images(self, noise: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Generates synthetic images using the trained generator model.
        Args:
            noise (np.ndarray): Latent vector for the generator input.
            labels (np.ndarray): Labels for conditioning the generation.
        Returns:
            np.ndarray: Generated images.
        """
        if noise.shape[0] != labels.shape[0]:
            raise ValueError("The number of noise samples must match the number of labels.")
        return self.generator.predict([noise, labels])


class ConditionalGANDiscriminator:
    def __init__(self, img_shape: Tuple[int, int, int], num_classes: int = 4, db_manager=None):
        """
        Initializes the Conditional GAN Discriminator class.
        Args:
            img_shape (Tuple[int, int, int]): Shape of the input image (height, width, channels).
            num_classes (int): Number of classes for conditional classification.
        """
        if not isinstance(img_shape, tuple) or len(img_shape) != 3:
            raise ValueError("img_shape must be a tuple of three integers.")
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.db_manager = db_manager
        self.discriminator = self._build_discriminator()

    def _build_discriminator(self) -> models.Model:
        """
        Builds and returns the GAN discriminator model conditioned on class labels.
        """
        # Image input
        img_input = tf.keras.Input(shape=self.img_shape, name="Image_Input")
        # Label input
        label_input = tf.keras.Input(shape=(1,), name="Label_Input")
        # Embed label input
        label_embedding = layers.Embedding(input_dim=self.num_classes, output_dim=np.prod(self.img_shape))(label_input)
        label_embedding = layers.Reshape(self.img_shape)(label_embedding)
        # Concatenate image and label embedding
        merged_input = layers.Concatenate()([img_input, label_embedding])
        # Convolutional layers to classify real or fake
        x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(merged_input)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs=[img_input, label_input], outputs=x, name="Conditional_Discriminator")
        model.summary()  # Print the model summary for better transparency
        return model


class ConditionalGAN:
    def __init__(self, latent_dim: int = 100, num_classes: int = 4, img_shape: Tuple[int, int, int] = (28, 28, 3)):
        """
        Initializes the Conditional GAN class.
        Args:
            latent_dim (int): Dimensionality of the latent space (noise input).
            num_classes (int): Number of gesture classes (labels).
            img_shape (Tuple[int, int, int]): Shape of the input image (height, width, channels).
        """
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.db_manager = DatabaseManager()
        self.generator = ConditionalGANGenerator(latent_dim, num_classes, img_shape, db_manager=self.db_manager)
        self.discriminator = ConditionalGANDiscriminator(img_shape, num_classes, db_manager=self.db_manager)
        # Make discriminator non-trainable when training the GAN
        self.discriminator.discriminator.trainable = False
        # Build the GAN model
        self.gan = self._build_gan()

    def _build_gan(self) -> models.Model:
        """
        Builds the full GAN model (generator + discriminator).
        """
        noise_input = tf.keras.Input(shape=(self.latent_dim,), name="Noise_Input")
        label_input = tf.keras.Input(shape=(1,), name="Label_Input")
        # Generate synthetic image from generator
        generated_image = self.generator.generator([noise_input, label_input])
        # Classify the generated image with the discriminator
        validity = self.discriminator.discriminator([generated_image, label_input])
        model = models.Model(inputs=[noise_input, label_input], outputs=validity, name="Conditional_GAN")
        model.summary()  # Print the model summary for better transparency
        return model

    def compile(self, optimizer: Adam):
        """
        Compiles the GAN model with the given optimizer.
        Args:
            optimizer (Adam): The optimizer to use for training.
        """
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    def load_real_data_from_db(self):
        """
        Fetches real gesture data from the database.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Real images and their corresponding labels.
        """
        try:
            gestures = self.db_manager.fetch_all_gestures()
            real_images = []
            real_labels = []

            for gesture in gestures:
                gesture_id, name, command, coordinates = gesture
                # Convert coordinates JSON string into a NumPy array
                coordinates_array = np.array(json.loads(coordinates)["coordinates"])
                real_images.append(coordinates_array)
                real_labels.append(gesture_id)

            real_images = np.array(real_images)
            real_labels = np.array(real_labels)
            return real_images, real_labels
        except Exception as e:
            print(f"Error loading real data from the database: {e}")
            return None, None

    def train(self, epochs: int = 10000, batch_size: int = 64,
              sample_interval: int = 1000, early_stopping_patience: int = 10, checkpoint_dir: str = './checkpoints',
              augment_data_with_gan: bool = True):
        """
        Trains the Conditional GAN.
        Args:
            epochs (int): Number of epochs to train.
            batch_size (int): Size of each batch.
            sample_interval (int): Interval to save images or display results.
            early_stopping_patience (int): Early stopping patience.
            checkpoint_dir (str): Directory to save model checkpoints.
            augment_data_with_gan (bool): Whether to augment data using the GAN generator.
        """
        # Load real gesture data from the database
        real_images, real_labels = self.load_real_data_from_db()
        if real_images is None or real_labels is None:
            raise ValueError("Failed to load real gesture data from the database.")

        half_batch = batch_size // 2
        # Adversarial ground truths
        valid = np.ones((half_batch, 1))
        fake = np.zeros((half_batch, 1))

        # Callbacks for early stopping and model checkpoint
        early_stopping = EarlyStopping(monitor='loss', patience=early_stopping_patience, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(checkpoint_dir, save_best_only=True, save_weights_only=True)

        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, real_images.shape[0], half_batch)
            real_imgs = real_images[idx]
            labels = real_labels[idx]

            # Optionally augment data with GAN-generated images
            if augment_data_with_gan:
                noise = np.random.normal(0, 1, (half_batch, self.latent_dim))
                gen_imgs = self.generator.generate_synthetic_images(noise, labels)
                real_imgs = np.concatenate([real_imgs, gen_imgs], axis=0)
                labels = np.concatenate([labels, labels], axis=0)

            d_loss_real = self.discriminator.discriminator.train_on_batch([real_imgs, labels], valid)
            d_loss_fake = self.discriminator.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator (via GAN model)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch([noise, labels], valid)

            # Optionally save the model and display results every `sample_interval` epochs
            if epoch % sample_interval == 0:
                print(f"Epoch: {epoch} | D Loss: {d_loss[0]} | G Loss: {g_loss}")
                # Save the model and/or results
                self.generator.generator.save(f'generator_epoch_{epoch}.h5')

if __name__ == "__main__":
    img_shape = (28, 28, 3)  # Example image shape (28x28 RGB)
    cgan = ConditionalGAN(latent_dim=100, num_classes=4, img_shape=img_shape)
    cgan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    cgan.train(epochs=10000, batch_size=64, sample_interval=1000, checkpoint_dir='./checkpoints', augment_data_with_gan=True)
