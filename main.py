import tensorflow as tf
import pandas as pd
from keras.optimizers import Adam
import matplotlib.pyplot as plt 
from models import *
from utils import *

def train(n_epochs, batch_size, codings_size, d_steps, gp_w):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    dataset = prepare_data(batch_size=batch_size)

    gen_projection_dim = 64
    gen_num_heads = 4
    gen_mlp_dim = [512, 512, 512]
    noise_dim=100

    generator = build_generator(noise_dim, gen_projection_dim, gen_num_heads, gen_mlp_dim)
    print(generator.summary())

    discriminator = build_discriminator()
    print(discriminator.summary())

    gan = WGAN(
        discriminator=discriminator, generator=generator,
        latent_dim=codings_size, discriminator_extra_steps=d_steps, gp_weight=gp_w
    )
    gan.compile(
        d_optimizer=Adam(learning_rate=0.0001),
        g_optimizer=Adam(learning_rate=0.0001),
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )

    visualization_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: visualize_generated_images(epoch, generator))
    history = gan.fit(dataset, epochs=n_epochs, verbose=1, callbacks=[visualization_callback, model_checkpoint_callback_weights])

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.set_title(f'Learning Curve', fontsize=18)
    pd.DataFrame(history.history).plot(ax=ax)
    ax.grid()

    generator.save(f'CUB-WGAN-GP_final.keras')

def main():
    BATCH_SIZE = 128
    CODINGS_SIZE = 100
    N_EPOCHS = 500
    D_STEPS = 5
    GP_WEIGHT = 10.0

    if len(tf.config.list_physical_devices('GPU')) != 0:  
        with tf.device("GPU:0"):
            print("Training on GPU")
            train(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, codings_size=CODINGS_SIZE, d_steps=D_STEPS, gp_w=GP_WEIGHT)
    else:
        print("NO GPU DETECTED! Training on CPU")
        train(n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, codings_size=CODINGS_SIZE, d_steps=D_STEPS, gp_w=GP_WEIGHT)
    
    print("Training Completed!!")

if __name__ == '__main__':
    main()