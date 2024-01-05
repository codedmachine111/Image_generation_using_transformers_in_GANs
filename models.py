import tensorflow as tf
from keras.layers import (Dense, Reshape, BatchNormalization, Conv2DTranspose, 
                          Dropout, LayerNormalization, Embedding, Input, Conv2D, LeakyReLU, Flatten,
                          Concatenate, concatenate, Lambda, ReLU) 
from utils import *

def scaled_dot_product(q, k, v):
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_qk = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
    attn_weights = tf.nn.softmax(scaled_qk, axis=-1)
    output = tf.matmul(attn_weights, v)
    return output

class MultiHeadAttentionL(tf.keras.layers.Layer):
    def __init__(self, model_dim, n_heads, initializer='glorot_uniform'):
        super(MultiHeadAttentionL, self).__init__()
        self.n_heads = n_heads
        self.model_dim = model_dim

        assert model_dim % self.n_heads == 0

        self.depth = model_dim // self.n_heads

        self.wq = Dense(model_dim, kernel_initializer=initializer)
        self.wk = Dense(model_dim, kernel_initializer=initializer)
        self.wv = Dense(model_dim, kernel_initializer=initializer)

        self.dense = Dense(model_dim, kernel_initializer=initializer)

    def split_into_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)

        scaled_attention = scaled_dot_product(q, k, v)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        original_size_attention = tf.reshape(scaled_attention, (batch_size, -1, self.model_dim))

        output = self.dense(original_size_attention)
        return output
    
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, n_patches, model_dim, initializer='glorot_uniform'):
        super(PositionalEmbedding, self).__init__()
        self.n_patches = n_patches
        self.position_embedding = Embedding(
            input_dim=n_patches, output_dim=model_dim,
            embeddings_initializer=initializer
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.n_patches, delta=1)
        return patches + self.position_embedding(positions)
    
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, rate=0.1, initializer='glorot_uniform', eps=1e-6):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttentionL(embed_dim, num_heads, initializer=initializer)
        self.ffn = tf.keras.Sequential(
            [Dense(mlp_dim, activation="relu", kernel_initializer=initializer),
             Dense(embed_dim, kernel_initializer=initializer),]
        )
        self.layernorm1 = LayerNormalization(epsilon=eps)
        self.layernorm2 = LayerNormalization(epsilon=eps)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_generator(noise_dim,
                    projection_dim,
                    num_heads,
                    mlp_dim):
    # Input layer
    embed_input = Input(shape=(1024,))
    x = Dense(256)(embed_input)
    mean_logsigma = LeakyReLU(alpha=0.2)(x)

    c = Lambda(generate_c)(mean_logsigma)

    noise_input = Input(shape=(noise_dim,))

    gen_input = Concatenate(axis=1)([c, noise_input])

    x = Dense(8 * 8 * projection_dim)(gen_input)
    x = Reshape((8 *8, projection_dim))(x)
    # x = layers.BatchNormalization()(x)

    positional_embeddings  = PositionalEmbedding(64, projection_dim)
    x = positional_embeddings(x)
    x = TransformerBlock(projection_dim, num_heads, mlp_dim[0])(x)
    x = TransformerBlock(projection_dim, num_heads, mlp_dim[0])(x)

    x = Reshape((8, 8, projection_dim))(x)

    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding="SAME", activation="selu")(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding="SAME", activation="selu")(x)
    x = BatchNormalization()(x)

    outputs = Conv2DTranspose(3, kernel_size=3, strides=2, padding="SAME",activation="tanh")(x)

    return tf.keras.Model(inputs=[embed_input,noise_input], outputs=outputs, name='generator')

def build_discriminator():
      image_input = Input(shape=(64,64,3))

      x = Conv2D(64, kernel_size=4, strides=2, padding="SAME", activation=LeakyReLU(0.2))(image_input)
      x = LayerNormalization()(x)
      x = Conv2D(128, kernel_size=4, strides=2, padding="SAME", activation=LeakyReLU(0.2))(x)
      x = LayerNormalization()(x)
      x = Conv2D(256, kernel_size=4, strides=2, padding="SAME", activation=LeakyReLU(0.2))(x)
      x = LayerNormalization()(x)
      x = Conv2D(512, kernel_size=4, strides=2, padding="SAME", activation=LeakyReLU(0.2))(x)

      x = Dropout(0.4)(x)

      embedding_input = Input(shape=(1024,))
      compressed_embedding = Dense(128)(embedding_input)
      compressed_embedding = ReLU()(compressed_embedding)

      compressed_embedding = tf.reshape(compressed_embedding, (-1, 1, 1, 128))
      compressed_embedding = tf.tile(compressed_embedding, (1, 4, 4, 1))

      concat_input = concatenate([x, compressed_embedding])

      x = Conv2D(64 * 8, kernel_size=1, strides=1, padding="SAME", activation=LeakyReLU(0.2))(concat_input)
      x = LayerNormalization()(x)

      x = Dropout(0.4)(x)
      x = Flatten()(x)

      outputs = Dense(1)(x)

      return tf.keras.Model(inputs=[image_input,embedding_input], outputs=outputs, name='discriminator')

class WGAN(tf.keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=5,
        gp_weight=10.0,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images, text_embeddings):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated, text_embeddings], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, dataset):

        real_images, text_embeddings = dataset

        if isinstance(real_images, tuple):
            real_images = real_images[0]
        if isinstance(text_embeddings, tuple):
            text_embeddings = text_embeddings[0]

        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator.
        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                fake_images = self.generator([text_embeddings,random_latent_vectors], training=True)
                fake_logits = self.discriminator([fake_images, text_embeddings], training=True)
                real_logits = self.discriminator([real_images, text_embeddings], training=True)
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images, text_embeddings)
                d_loss = d_cost + gp * self.gp_weight
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator([text_embeddings, random_latent_vectors], training=True)
            gen_img_logits = self.discriminator([generated_images, text_embeddings], training=True)
            g_loss = self.g_loss_fn(gen_img_logits)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}