from tensorflow.keras.layers import *
from tensorflow import keras
from tensorflow.keras import layers, models, Model
import tensorflow as tf
import numpy as np
import qkeras
from qkeras import *


quantizer = quantized_bits(16, 6, alpha=1)
quantized_relu = 'quantized_relu(16, 6, negative_slope=0.125)'
# if negative_slope: assert np.mod(np.log2(negative_slope), 1) == 0

########### plain ae model

def build_autoencoder(input_dim=57, q=True):
    if q is False:
        x_in = layers.Input(shape=(input_dim,), name='in')
        #x = layers.BatchNormalization(name='bn1')(x_in)
        x = layers.Dense(32, name='dense1')(x_in)
        x = layers.LeakyReLU(alpha=0.1, name='act1')(x)
        x = layers.Dense(16, name='dense2')(x)
        x = layers.LeakyReLU(alpha=0.1, name='act2')(x)
        x = layers.Dense(3, name='dense3')(x)
        x = layers.LeakyReLU(alpha=0.1, name='act3')(x)
        
        x = layers.Dense(16, name='dense4')(x)
        x = layers.LeakyReLU(alpha=0.1, name='act4')(x)
        x = layers.Dense(32, name='dense5')(x)
        x = layers.LeakyReLU(alpha=0.1, name='act5')(x)
        x = layers.Dense(input_dim, name='dense6')(x)
    
    else:
        x_in = layers.Input(shape=(input_dim,), name='in')
        #x = QBatchNormalization(beta_quantizer=quantizer, gamma_quantizer=quantizer, mean_quantizer=quantizer, variance_quantizer=quantizer, name='qbn1')(x_in)
        x = QDense(32, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qdense1')(x_in)
        x = QActivation(quantized_relu, name='qact1')(x)
        x = QDense(16, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qdense2')(x)
        x = QActivation(quantized_relu, name='qact2')(x)
        x = QDense(3, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qdense3')(x)
        x = QActivation(quantized_relu, name='qact3')(x)
        
        x = QDense(16, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qdense4')(x)
        x = QActivation(quantized_relu, name='qact4')(x)
        x = QDense(32, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qdense5')(x)
        x = QActivation(quantized_relu, name='qact5')(x)
        x = QDense(input_dim, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qdense6')(x)
    
    model = Model(x_in, x, name='ae')
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss='mse')
    return model


def compute_mse_for_ae(model, x):
    x_reco = model.predict(x)
    mse = np.mean(np.square(x - x_reco), axis=1)
    return mse

########### vae model

def vae_sampling(z_par):
    z_mean, z_log_var = z_par
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    z_sampled = z_mean + tf.exp(0.5 * z_log_var) * epsilon
    return z_sampled


def build_vae(input_dim=57, beta=0.5, q=True):
    if q is False:
        encoder_in = layers.Input(shape=(input_dim,), name='encoder_in')
        #x = layers.BatchNormalization(name='bn1')(encoder_in)
        x = layers.Dense(32, name='dense1')(encoder_in)
        x = layers.LeakyReLU(alpha=0.1, name='act1')(x)
        x = layers.Dense(16, name='dense2')(x)
        x = layers.LeakyReLU(alpha=0.1, name='act2')(x)
    
        z_mean = layers.Dense(3, name='z_mean')(x)
        z_log_var = layers.Dense(3, name='z_log_var')(x)
    else:
        encoder_in = layers.Input(shape=(input_dim,), name='encoder_in')
        #x = QBatchNormalization(beta_quantizer=quantizer, gamma_quantizer=quantizer, mean_quantizer=quantizer, variance_quantizer=quantizer, name='qbn1')(encoder_in)
        x = QDense(32, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qdense1')(encoder_in)
        x = QActivation(quantized_relu, name='qact1')(x)
        x = QDense(16, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qdense2')(x)
        x = QActivation(quantized_relu, name='qact2')(x)

        z_mean = QDense(3, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qz_mean')(x)
        z_log_var = QDense(3, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qz_log_var')(x)

    z = layers.Lambda(vae_sampling, output_shape=(3,), name='z_sampling')([z_mean, z_log_var])
    
    if q is False:
        decoder_in = layers.Input(shape=(3,), name='decoder_in')
        xx = layers.Dense(16, name='dense4')(decoder_in)
        xx = layers.LeakyReLU(alpha=0.1, name='act4')(xx)
        xx = layers.Dense(32, name='dense5')(xx)
        xx = layers.LeakyReLU(alpha=0.1, name='act5')(xx)
        decoder_out = layers.Dense(input_dim, name='dense6')(xx)
    else:
        decoder_in = layers.Input(shape=(3,), name='decoder_in')
        xx = QDense(16, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qdense4')(decoder_in)
        xx = QActivation(quantized_relu, name='qact4')(xx)
        xx = QDense(32, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qdense5')(xx)
        xx = QActivation(quantized_relu, name='qact5')(xx)
        decoder_out = QDense(input_dim, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qdense6')(xx)

    encoder = Model(encoder_in, [z_mean, z_log_var], name='vae_encoder')
    decoder = Model(decoder_in, decoder_out, name='vae_decoder')
    
    reco = decoder(z)
    vae = Model(encoder_in, reco, name='vae')

    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    kl_loss = tf.reduce_mean(kl_loss)
    
    vae.add_loss(beta * kl_loss)
    
    def reco_loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        return (1 - beta) * mse
    
    vae.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss=reco_loss)

    return vae, encoder, decoder


def compute_kl_for_vae(encoder, x):
    z_mean, z_log_var = encoder.predict(x)
    kl = -0.5 * np.sum(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=1)
    return kl


def compute_r_for_vae(encoder, x):
    z_mean, z_log_var = encoder.predict(x)
    sigma = np.exp(0.5 * z_log_var)
    r_i = z_mean / (sigma + 1e-6)
    r = np.sum(np.square(r_i), axis=1)
    return r


########### nf model

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
    def __init__(self, num_flows, q, **kwargs):
        super(NormalizingFlowModel, self).__init__(**kwargs)
        if q is False:
            #self.bn1 = layers.BatchNormalization(name='bn1')
            self.dense1 = layers.Dense(32, name='dense1')
            self.act1 = layers.LeakyReLU(alpha=0.1, name='act1')
            self.dense2 = layers.Dense(16, name='dense2')
            self.act2 = layers.LeakyReLU(alpha=0.1, name='act2')
            self.dense3 = layers.Dense(4, name='dense3')
            #self.act3 = layers.LeakyReLU(alpha=0.1, name='act3')
        else:
            #self.bn1 = QBatchNormalization(beta_quantizer=quantizer, gamma_quantizer=quantizer, mean_quantizer=quantizer, variance_quantizer=quantizer, name='qbn1')
            self.dense1 = QDense(32, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qdense1')
            self.act1 = QActivation(quantized_relu, name='qact1')
            self.dense2 = QDense(16, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qdense2')
            self.act2 = QActivation(quantized_relu, name='qact2')
            self.dense3 = QDense(4, kernel_quantizer=quantizer, bias_quantizer=quantizer, name='qdense3')
            #self.act3 = QActivation(quantized_relu, name='qact3')

        self.flows = [PlanarFlow() for _ in range(num_flows)]
    
    def call(self, inputs):
        #x = self.bn1(inputs)
        x = self.dense1(inputs)
        x = self.act1(x)
        x = self.dense2(x)
        x = self.act2(x)
        x = self.dense3(x)
        #x = self.act3(x)

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


