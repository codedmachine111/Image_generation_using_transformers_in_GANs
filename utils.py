import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

def generate_c(x):
    mean = x[:, :128]
    log_sigma = x[:, 128:]

    stddev = K.exp(log_sigma)
    epsilon = K.random_normal(shape=K.constant((mean.shape[1],), dtype='int32'))
    c = stddev * epsilon + mean

    return c

def show_dataset_images(images):
    fig, axes = plt.subplots(4,4, figsize=(8,8))
    axes = axes.ravel()
    for i in range(16):
      axes[i].imshow(images[i])
      axes[i].axis('off')
    plt.show()

# Visualization callback
def visualize_generated_images(epoch, generator, dataset, latent_dim=100, num_samples=5):
    real_images, text_embeddings = next(iter(dataset.take(1)))
    random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
    generated_images = generator.predict([text_embeddings[:num_samples], random_latent_vectors])
    generated_images += 1
    generated_images *= 127.5

    # Display the generated images (adjust this based on your needs)
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
    for i in range(num_samples):
        axes[i].imshow(generated_images[i].astype(np.uint8))
        axes[i].axis('off')
    plt.suptitle(f'Generated Images - Epoch {epoch}')
    plt.show()

def prepare_data(batch_size, data_path):
    x_train_path = data_path + "/X_train_CUB.npy"
    embed_train_path = data_path + "/embeddings_train_CUB.npy"

    x_train_64 = np.load(x_train_path)
    embed_train_64 = np.load(embed_train_path)

    print(f'Dataset images shape: {x_train_64.shape}\n')
    print(f'Text embeddings shape: {embed_train_64.shape}\n')

    print(f'Dataset images: \n')
    show_dataset_images(x_train_64)

    # Normalization
    x_train = x_train_64.astype(np.float32) / 127.5
    x_train = x_train - 1

    dataset = tf.data.Dataset.from_tensor_slices((x_train, embed_train_64))
    dataset = dataset.shuffle(1024)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)   

    return dataset 

def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)