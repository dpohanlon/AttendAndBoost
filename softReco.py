from keras import backend as K
import keras
from keras.layers import Layer

import tensorflow as tf

from keras import backend as K

class Attention(Layer):
    def __init__(self, units):

        self.units = units

        super(Attention, self).__init__()

    def build(self, input_shape):

        # Need to know input shapes in order to register
        # number of parameters (weights)

        # These undergo transformations in call(), so can't
        # just build these here and request input_shape
        # and output_shape

        self.W1 = keras.layers.Dense(self.units, activation = 'relu', use_bias = True)
        self.W2 = keras.layers.Dense(self.units, activation = 'relu', use_bias = True)
        self.V = keras.layers.Dense(1, activation = None, use_bias = True)

        super(Attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        #Neccessary - automatic inference, gets the shape wrong!

        shape_hidden, shape_values = input_shape
        batch_size = shape_hidden[0]

        shape_context = (batch_size, shape_hidden[1])
        shape_attention_weights = (batch_size, shape_values[1], 1)

        return [shape_context, shape_attention_weights]

    def compute_mask(self, inputs, mask = None):
        return mask

    def count_params(self):
        # A placeholder
        return 42

    def call(self, input):
        query, values = input

        # query = K.print_tensor(query, 'query')
        # values = K.print_tensor(values, 'values')

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = K.expand_dims(query, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(K.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # score = K.print_tensor(score, 'score')

        # attention_weights shape == (batch_size, max_length, 1) # 32?
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = K.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = K.sum(context_vector, axis=1)

        attention_weights = K.print_tensor(attention_weights, 'attention_weights')

        return [context_vector, attention_weights]

class SoftRecoLayer(Layer):

    def __init__(self, units, **kwargs):
        self.units = units

        self.I = tf.eye(4)
        self.epsilon = 1E-9

        u = tf.constant([[-1, 0, 0, 0]] + 3 * [[0, -1, -1, -1]], tf.float32)

        # Reshape to add batch dimension
        self.u = tf.reshape(u, (1, 4, 4))

        self.attention = Attention(self.units)

        super(SoftRecoLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        super(SoftRecoLayer, self).build(input_shape)

    def call(self, inputs):

        query, values, fourVectorInputs = inputs

        attention_outputs, attention_weights = self.attention([query, values])

        # attention_outputs = K.print_tensor(attention_outputs, 'attention_outputs')
        # attention_weights = K.print_tensor(attention_weights, 'attention_weights')

        restFrame4Vectors = self.calculateWeightedRestFrames(attention_weights, fourVectorInputs)

        # restFrame4Vectors = K.print_tensor(restFrame4Vectors, 'restFrame4Vectors')

        boosts = self.calculateBoosts(restFrame4Vectors)

        fourVectorOutputs = self.applyBoost(boosts, fourVectorInputs)

        return [attention_outputs, attention_weights, fourVectorOutputs]

    def calculateWeightedRestFrames(self, weights, fourVectors):

        # Broadcast weights so that the weight is 'tiled' across each 4v component

        weights = tf.broadcast_to(weights, tf.shape(fourVectors))

        return tf.einsum('Bij, Bij -> Bj', weights, fourVectors)

    def applyBoost(self, boost, fourVectors):

        # (batch, 4, 4) . (batch, n4v, 4) -> (batch, n4v, 4)
        return tf.einsum('Bij, Bki -> Bkj', boost, fourVectors)

    def calculateBoosts(self, restFrame4Vector):

        # First element (slice to maintain last axis)
        E = restFrame4Vector[:,:1]

        # Last three elements
        p = restFrame4Vector[:,1:]

        beta = p / E

        absBeta = tf.expand_dims(tf.sqrt(tf.reduce_sum(beta ** 2, axis = -1)), -1)

        n = beta / absBeta

        gamma = 1. / tf.sqrt(1. - absBeta ** 2 + self.epsilon)

        e = tf.concat([tf.ones_like(E, tf.float32), -n], -1)

        l1 = self.u + tf.reshape(gamma, (-1, 1, 1))
        l2 = (self.u + 1) * tf.reshape(absBeta, (-1, 1, 1)) - self.u
        l3 = tf.einsum('Bi, Bj -> Bij', e, e)

        lamb = self.I + l1 * l2 * l3

        return lamb

    def compute_output_shape(self, input_shape):

        attentionOutputShape = self.attention.compute_output_shape(input_shape[:-1])

        return attentionOutputShape + [input_shape[-1]]

    def compute_mask(self, inputs, mask = None):

        # Attention outputs has no mask
        # Attention weights has same mask as input to attention
        # Four vectors have same mask as input to attention

        return [None, mask[-1], mask[-1]]

def testAttention():

    inputs = keras.layers.Input((10, 18))
    tracks = keras.layers.Masking(mask_value = -999, name = 'mask')(inputs)

    tracks, hidden = keras.layers.GRU(3, activation = 'relu', return_state = True, return_sequences = True)(tracks)
    context_vector, attention_weights = Attention(128)([hidden, tracks])

    out = keras.layers.Dense(1)(context_vector)

    model = keras.models.Model(inputs = inputs, outputs = out)

    model.summary(line_length = 150)

    testInput = np.random.normal(0, 1, size = (123, 10, 18))
    testTargets = np.random.randint(0, 2, size = 123)

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit(testInput, testTargets, epochs = 1)

def testReco():

    inputFeatures = keras.layers.Input((10, 18))
    inputFourVectors = keras.layers.Input((10, 4))

    inputFeaturesMasked = keras.layers.Masking(mask_value = -999, name = 'maskFeatures')(inputFeatures)
    inputFourVectorsMasked = keras.layers.Masking(mask_value = -999, name = 'maskFourVectors')(inputFourVectors)

    tracks, hidden = keras.layers.GRU(1, activation = 'relu', return_state = True, return_sequences = True)(inputFeaturesMasked)
    attention_outputs, attention_weights, fourVectorOutputs = SoftRecoLayer(32)([hidden, tracks, inputFourVectorsMasked])

    tracks = keras.layers.Concatenate()([tracks, fourVectorOutputs])

    tracks = keras.layers.GRU(1, activation = 'relu', return_state = False, return_sequences = False)(tracks)

    out = keras.layers.Dense(1)(tracks)

    model = keras.models.Model(inputs = [inputFeatures, inputFourVectors], outputs = out)

    model.summary(line_length = 200)

    testTargets = np.random.randint(0, 2, size = 123)
    testInputFeatures = np.random.normal(0, 1, size = (123, 10, 18))

    import uproot
    import pandas as pd

    dataFile = '/Users/MBP/GoogleDrive/Bu2pipipi_sqDalitz_withVars.h5'
    data = pd.read_hdf(dataFile)

    fourVectorsInput = data[['pip_0_E', 'pip_0_PX', 'pip_0_PY', 'pip_0_PZ']][:123 * 10].values
    fourVectorsInput = fourVectorsInput.reshape((123, 10, 4))

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    model.fit([testInputFeatures, fourVectorsInput], testTargets, epochs = 100)

if __name__ == '__main__':

    import numpy as np

    from pprint import pprint

    testReco()
    # testAttention()
