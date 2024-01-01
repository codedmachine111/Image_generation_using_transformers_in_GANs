import numpy as np
import tensorflow as tf
import pickle
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

def plot_results(images, n_cols=None, title=None):

    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1

    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)

    fig = plt.figure(figsize=(n_cols, n_rows))

    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image.astype(np.uint8), cmap="binary")
        plt.axis("off")

    plt.suptitle(title)

def show_dataset_images(images):
    fig, axes = plt.subplots(4,4, figsize=(8,8))
    axes = axes.ravel()
    for i in range(16):
      axes[i].imshow(images[i])
      axes[i].axis('off')
    plt.show()

def visualize_generated_images(epoch, generator, latent_dim=100, num_samples=5):
    random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
    generated_images = generator.predict(random_latent_vectors)
    generated_images += 1
    generated_images *= 127.5

    # Display the generated images (adjust this based on your needs)
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
    for i in range(num_samples):
        axes[i].imshow(generated_images[i].astype(np.uint8))
        axes[i].axis('off')
    plt.suptitle(f'Generated Images - Epoch {epoch}')
    plt.show()

# Define the file path for saving the model
model_checkpoint_path_weights = 'ckpts/CUB-WGAN-GP-weights-{epoch:02d}.keras'

# Create a ModelCheckpoint callback
model_checkpoint_callback_weights = ModelCheckpoint(
    filepath=model_checkpoint_path_weights,
    save_freq='epoch',  # Save every epoch
    save_weights_only=True,  # Save only the weights
)

def load_images(path):
    #Loading images from pickle file
    with open(path, 'rb') as f_in:
        images = pickle.load(f_in)
    return images

def load_data(pickle_data_file):
    #Load images and embeddings
    x = np.array(load_images(pickle_data_file))
    return x

def prepare_data(batch_size):
    pickle_path_64 = "data/64images.pickle"
    x_train_64 = load_data(pickle_path_64)

    print(f'Dataset images shape: {x_train_64.shape}\n')
    print(f'Dataset images: \n')
    show_dataset_images(x_train_64)

    # Normalization
    x_train = x_train_64.astype(np.float32) / 127.5
    x_train = x_train - 1

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)   

    return dataset 
