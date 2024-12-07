import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
import random

# Ensure clean TensorFlow session
tf.keras.backend.clear_session()


# Prepare numeric data
def prepare_gan_data(pokemon_data, numeric_cols):
    return pokemon_data[numeric_cols].values


# Build GAN generator
def build_generator(latent_dim, data_dim):
    model = Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(alpha=0.2),
        Dense(256),
        LeakyReLU(alpha=0.2),
        Dense(data_dim, activation='linear')
    ])
    return model


# Build GAN discriminator
def build_discriminator(data_dim):
    model = Sequential([
        Dense(256, input_dim=data_dim),
        LeakyReLU(alpha=0.2),
        Dense(128),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')
    ])
    return model


# GAN training
def train_gan(generator, discriminator, gan_data, latent_dim, epochs=1000, batch_size=32):
    generator_optimizer = Adam(0.0002, 0.5)
    discriminator_optimizer = Adam(0.0002, 0.5)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    for epoch in range(epochs):
        # Train Discriminator
        noise = tf.random.normal((batch_size, latent_dim))
        generated_data = generator(noise)
        idx = np.random.randint(0, gan_data.shape[0], batch_size)
        real_data = tf.convert_to_tensor(gan_data[idx])

        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            real_predictions = discriminator(real_data)
            fake_predictions = discriminator(generated_data)
            real_loss = loss_fn(real_labels, real_predictions)
            fake_loss = loss_fn(fake_labels, fake_predictions)
            d_loss = real_loss + fake_loss
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # Train Generator
        with tf.GradientTape() as tape:
            noise = tf.random.normal((batch_size, latent_dim))
            generated_data = generator(noise)
            predictions = discriminator(generated_data)
            g_loss = loss_fn(real_labels, predictions)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        # Log
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D loss = {d_loss.numpy()}, G loss = {g_loss.numpy()}")


# Prepare name data
def prepare_name_data(pokemon_names):
    tokenizer = Tokenizer(char_level=True, lower=True)
    tokenizer.fit_on_texts(pokemon_names)

    sequences = []
    for name in pokemon_names:
        tokenized_name = tokenizer.texts_to_sequences([name])[0]
        for i in range(1, len(tokenized_name)):
            sequences.append(tokenized_name[:i + 1])

    max_seq_length = max(len(seq) for seq in sequences)
    sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_length, padding='pre')
    X_name = sequences[:, :-1]
    y_name = to_categorical(sequences[:, -1], num_classes=len(tokenizer.word_index) + 1)

    return X_name, y_name, tokenizer, len(tokenizer.word_index) + 1, max_seq_length


# Build name generator
def build_name_generator(vocab_size, max_seq_length):
    model = Sequential([
        Embedding(vocab_size, 50, input_length=max_seq_length - 1),
        LSTM(100),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


# Generate names
def generate_name_with_randomness(model, tokenizer, max_length, start_char='a', temperature=1.0):
    tokenized_start = tokenizer.texts_to_sequences([start_char])[0]
    name = tokenized_start
    for _ in range(max_length):
        padded_name = tf.keras.preprocessing.sequence.pad_sequences([name], maxlen=max_length, padding='pre')
        predictions = model.predict(padded_name)[0]
        predictions = np.log(predictions + 1e-9) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
        next_char = np.random.choice(len(probabilities), p=probabilities)
        if next_char == 0:  # End of name
            break
        name.append(next_char)
    return ''.join(tokenizer.sequences_to_texts([name])[0])


# Pokémon Generator
def generate_pokemon(generator, name_generator, tokenizer, numeric_cols, latent_dim, num_samples=10):
    noise = tf.random.normal((num_samples, latent_dim))
    generated_stats = generator(noise).numpy()

    generated_names = [
        generate_name_with_randomness(name_generator, tokenizer, len(numeric_cols), start_char=random.choice('abcdefghijklmnopqrstuvwxyz'), temperature=1.2)
        for _ in range(num_samples)
    ]

    generated_pokemon_df = pd.DataFrame(generated_stats, columns=numeric_cols)
    generated_pokemon_df['name'] = generated_names
    return generated_pokemon_df


# Main Execution
if __name__ == "__main__":
    # Load data
    pokemon_data = pd.read_csv('pokemon_data.csv')
    numeric_cols = ["height", "weight", "base_experience", "hp", "attack", "defense", "special-attack", "special-defense", "speed"]
    pokemon_names = pokemon_data['name'].str.lower().tolist()
    gan_data = prepare_gan_data(pokemon_data, numeric_cols)
    latent_dim = 10

    # Prepare name data
    X_name, y_name, tokenizer, vocab_size, max_seq_length = prepare_name_data(pokemon_names)

    # Build and train name generator
    name_generator = build_name_generator(vocab_size, max_seq_length)
    name_generator.fit(X_name, y_name, epochs=50, batch_size=32)

    # Build GAN
    generator = build_generator(latent_dim, len(numeric_cols))
    discriminator = build_discriminator(len(numeric_cols))
    train_gan(generator, discriminator, gan_data, latent_dim, epochs=1000, batch_size=32)

    # Generate Pokémon
    generated_pokemon = generate_pokemon(generator, name_generator, tokenizer, numeric_cols, latent_dim, num_samples=10)
    print("Generated Pokémon with Stats and Names:")
    print(generated_pokemon)
