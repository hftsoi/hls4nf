from tensorflow.keras.layers import *
from tensorflow import keras
from tensorflow.keras import layers, models, Model
import tensorflow as tf
import numpy as np
import qkeras
from qkeras import *


def build_autoencoder(input_dim=57):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(32, activation='relu')(input_layer)
    encoded = layers.Dense(16, activation='relu')(encoded)
    latent = layers.Dense(3, activation='relu')(encoded)
    
    decoded = layers.Dense(16, activation='relu')(latent)
    decoded = layers.Dense(32, activation='relu')(decoded)
    output_layer = layers.Dense(input_dim, activation='relu')(decoded)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.005), loss='mse')
    return model


def compute_mse_for_ae(model, x):
    reco = model.predict(x)
    mse = np.mean(np.square(x - reco), axis=1)
    return mse


class PlanarFlow(layers.Layer):
    def __init__(self, **kwargs):
        super(PlanarFlow, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.w = self.add_weight(name='w', shape=(self.input_dim,),
                                 initializer='random_normal', trainable=True)
        self.u = self.add_weight(name='u', shape=(self.input_dim,),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='b', shape=(1,),
                                 initializer='random_normal', trainable=True)
        super(PlanarFlow, self).build(input_shape)
    
    def call(self, inputs):
        # f(x) = x + u * h(w^T x + b)
        linear_term = tf.tensordot(inputs, self.w, axes=1) + self.b
        h = tf.tanh(linear_term)
        outputs = inputs + tf.expand_dims(h, -1) * self.u

        # log|det| = log|1 + u^T h'(w^T x + b) * w|
        h_prime = 1 - tf.square(tf.tanh(linear_term))
        psi = tf.expand_dims(h_prime, -1) * self.w
        u_dot_psi = tf.reduce_sum(self.u * psi, axis=-1)
        log_det = tf.math.log(tf.abs(1 + u_dot_psi) + 1e-6)

        return outputs, log_det
    

class NormalizingFlowModel(Model):
    def __init__(self, num_flows, **kwargs):
        super(NormalizingFlowModel, self).__init__(**kwargs)
        self.dense1 = layers.Dense(32, name='dense1')
        self.act1 = layers.LeakyReLU(alpha=0.1, name='act1')
        self.dense2 = layers.Dense(16, name='dense2')
        self.act2 = layers.LeakyReLU(alpha=0.1, name='act2')
        self.dense3 = layers.Dense(4, name='dense3')
        self.act3 = layers.LeakyReLU(alpha=0.1, name='act3')

        self.flows = [PlanarFlow() for _ in range(num_flows)]
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        x = self.dense3(x)
        x = self.act3(x)

        log_det_total = 0

        for flow in self.flows:
            x, log_det = flow(x)
            log_det_total += log_det

        return x, log_det_total
    

def base_log_prob(z):
    # gaussian base
    d = tf.cast(tf.shape(z)[-1], tf.float32)
    log_prob = -0.5 * (tf.reduce_sum(tf.square(z), axis=-1) + d * tf.math.log(2 * np.pi))
    return log_prob


# for training (average over batch)
def nll_loss(model_nf, x):
    z, log_det_total = model_nf(x)
    log_prob = base_log_prob(z) + log_det_total
    loss = -tf.reduce_mean(log_prob)
    return loss


# for inference (by event)
def compute_nll_for_nf(model_nf, data):
    z, log_det_total = model_nf(data)
    log_prob = base_log_prob(z) + log_det_total
    nll = -log_prob.numpy()
    return nll


