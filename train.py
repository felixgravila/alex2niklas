#%%

import json
import tensorflow as tf
import tensorflow.keras.layers as L
import random
import numpy as np
import editdistance as ed

with open("names.json", "r") as f:
    names = json.load(f)
with open("c_to_i.json", "r") as f:
    c_to_i = json.load(f)
with open("i_to_c.json", "r") as f:
    i_to_c = json.load(f)
    i_to_c = list(i_to_c.values())

END_TOK = len(i_to_c)
BATCH = 32

# compute longest possible name
max_input_len = max([len(n) for n in names])

# Add end token to dictionary
i_to_c.extend(["$"])
c_to_i.update({
    "$":END_TOK
})

def chars_to_numbers(x):
    """
    Transform string name to list of ints
    """
    return [c_to_i[c] for c in x]

def pad_to(name):
    """
    Pad name to required length
    """
    n = name + [END_TOK]*(max_input_len-len(name))
    return n

def ohe_to_name(ohe):
    """
    Transform OHE output of model back into the name, removing $
    """
    deohe = tf.argmax(ohe, axis=1)
    locs = deohe<26
    wordnums = deohe[locs]
    word = "".join([i_to_c[w] for w in wordnums.numpy()])
    return word

def ohe_to_name_with_symbols(ohe):
    """
    Transform OHE output of model back into the name, keeping $
    """
    deohe = tf.argmax(ohe, axis=1)
    word = "".join([i_to_c[w] for w in deohe.numpy()])
    return word


#%%

# transform names to numbers
# then pad them to max_input_len
# then one-hot encode
names_ohe = [tf.one_hot(pad_to(chars_to_numbers(n)), END_TOK+1) for n in names]

# %%

# Create VAE model
class VAE(tf.keras.Model):
    def __init__(self, neurons, vecsize, variance_weight=0.2, kl_loss_weight=0.2) -> None:
        super().__init__()

        # Cowboy reducing variance impact to get this to work faster for this toy project...
        self.variance_weight = variance_weight
        self.kl_loss_weight = kl_loss_weight

        self.encoder = tf.keras.Sequential([
            L.Input(shape=(None, END_TOK+1)),
            L.Conv1D(32, 3, padding="same"),
            L.Conv1D(32, 3, padding="same"),
            L.Conv1D(32, 3, padding="same"),
            L.Bidirectional(L.LSTM(neurons, return_sequences=True)),
            L.Bidirectional(L.LSTM(neurons, return_sequences=True)),
            L.LSTM(neurons)
        ])

        self.decoder = tf.keras.Sequential([
            L.Input(shape=(vecsize,)),
            L.RepeatVector(max_input_len),
            L.Bidirectional(L.LSTM(neurons, return_sequences=True)),
            L.Bidirectional(L.LSTM(neurons, return_sequences=True)),
            L.Conv1D(32, 3, padding="same"),
            L.Conv1D(32, 3, padding="same"),
            L.Conv1D(END_TOK+1, 3, padding="same")
        ])

        self.z_mean = L.Dense(vecsize)
        self.z_log_var = L.Dense(vecsize)

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

        self.rec_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, axis=-1)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def __call__(self, data, training=False):
        z_mean, z_log_var = self.encode(data)
        if training:
            encoded = self.sample((z_mean, z_log_var))
        else:
            encoded = z_mean
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, data):
        encoded = self.encoder(data)
        z_mean = self.z_mean(encoded)
        z_log_var = self.z_log_var(encoded)
        return z_mean, z_log_var

    def decode(self, data):
        return self.decoder(data)

    def sample(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon * self.variance_weight

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encode(data)
            z = self.sample([z_mean, z_log_var])
            reconstruction = self.decoder(z)
            reconstruction_loss = self.rec_loss(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            kl_loss = kl_loss * self.kl_loss_weight
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        z_mean, z_log_var = self.encode(data)
        reconstruction = self.decoder(z_mean)
        reconstruction_loss = self.rec_loss(data, reconstruction)
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        kl_loss = kl_loss * self.kl_loss_weight
        total_loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

vae = VAE(neurons=64, vecsize=256, variance_weight=0.3, kl_loss_weight=0.2)
vae.compile(optimizer=tf.keras.optimizers.Adam())

# %%

# Create datasets
random.shuffle(names_ohe)
train_gen = tf.data.Dataset.from_tensor_slices(names_ohe[:20000]).shuffle(1000).batch(BATCH)
val_gen = tf.data.Dataset.from_tensor_slices(names_ohe[20000:]).shuffle(1000).batch(BATCH)

igen = iter(val_gen)
x = next(igen)
print(ohe_to_name_with_symbols(x[0]))

# %%

class EDCallback(tf.keras.callbacks.Callback):
    """
    Custom callback that checks mean editdistance between input and predictions
    If mean editdistance is 0 for the batch of 32, model is good enough, stop training.
    """
    def __init__(self, test_data) -> None:
        super().__init__()
        self.test_data = test_data
        self.names = [ohe_to_name(x) for x in test_data]

    def on_epoch_end(self, epoch, logs=None):
        result = self.model(self.test_data)
        res_decoded = [ohe_to_name(x) for x in result]
        eds = np.array([ed.eval(a,b) for (a,b) in zip(self.names, res_decoded)])
        mean_eds = np.mean(eds)
        print((s := f", editdistance {mean_eds}, {sum(eds==0)}/{len(result)} ({sum(eds==0)/len(result)*100:.01f}%) perfect."))
        if mean_eds == 0:
            self.model.stop_training = True

#%%

# Train the model
vae.fit(train_gen, epochs=50, validation_data=val_gen, callbacks=[EDCallback(x)])

#%%

# Show some examples
x = next(igen)
res = vae(x)
for i,o in zip(x[:10],res):
    print(ohe_to_name(i), "->", ohe_to_name(o))

# %%


def prepare(name:str):
    """
    Function that takes a name string
    Returns one hot vector ready for the model
    """
    n = chars_to_numbers(name)
    n = pad_to(10)(n)
    n = tf.one_hot(n, END_TOK+1)
    n = tf.expand_dims(n, 0)
    return n

# Get encodings for Alex and Niklas
alex_encoded, _ = vae.encode(prepare("alex"))
niklas_encoded, _ = vae.encode(prepare("niklas"))

# Get intermediate names from 100% Alex to 100% Niklas
for perc in np.arange(0,1.1,0.1):
    wsum = perc*niklas_encoded + (1-perc)*alex_encoded
    decoded = vae.decode(wsum)[0]
    decoded = ohe_to_name(decoded)
    print(f"{round((1-perc)*100):03d}% alex + {round((perc)*100):03d}% niklas -> {decoded}")


# %%
