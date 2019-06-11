"""
Train a simple RNN model for Inclusive Flavour Tagging using track information.
- Using DataGenerator class to stream data to the GPU, to avoid storing loads of data in RAM
- No transformations on data are done in this part now - these are all done per-batch in DataGenerator
"""

__author__ = "Daniel O'Hanlon <daniel.ohanlon@cern.ch>"

from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(['fivethirtyeight', 'seaborn-whitegrid', 'seaborn-ticks'])
import matplotlib.ticker as plticker

import keras
from keras.callbacks import TerminateOnNaN, TensorBoard, EarlyStopping
from keras.optimizers import Adam, SGD

from pprint import pprint

# FIX ME
import sys
sys.path.append('../')

from utils.utils import decision_and_mistag, saveModel, exportForCalibration
from utils.plotUtils import makeTrainingPlots

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from softReco import SoftRecoLayer, Attention

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

# ------------------------------------------------------------
# needs to be defined as activation class otherwise error
# AttributeError: 'Activation' object has no attribute '__name__'
class Swish(keras.layers.Activation):

    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})
# ------------------------------------------------------------

import shelve

from dataGeneratorK import createSplitGenerators

def tagNetworkSoftReco(trackShape):

    trackInput = keras.layers.Input(trackShape, name = 'trackInput')

    tracks = keras.layers.Masking(mask_value = 0, name = 'maskFeatures')(trackInput)

    tracks = keras.layers.TimeDistributed(keras.layers.Dense(8, activation = 'relu'), name = 'td_dense1')(tracks)

    tracks, hidden = keras.layers.GRU(8, activation = 'relu', return_sequences = True, return_state = True, recurrent_dropout = 0.5, name = 'noseq_gru')(tracks)

    fourVecInput = keras.layers.Input((100, 4), name = 'fourVecInput')
    fourVectors = keras.layers.Masking(mask_value = 0, name = 'maskFourVectors')(fourVecInput)

    attention_outputs, attention_weights, fourVectorOutputs = SoftRecoLayer(8)([hidden, tracks, fourVectors])

#     fourVectorOutputs = keras.layers.BatchNormalization()(fourVectorOutputs)
    fourVectorOutputs = keras.layers.GRU(8, activation = 'relu', return_sequences = False, name = 'track_gru')(fourVectorOutputs)

    fourVectorOutputs = keras.layers.BatchNormalization()(fourVectorOutputs)

    tracks = keras.layers.Concatenate()([attention_outputs, fourVectorOutputs])

    tracks = keras.layers.Dense(8, activation = 'relu', name = 'out_dense_1')(tracks)

    outputTag = keras.layers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return keras.Model(inputs = [trackInput, fourVecInput], outputs = outputTag)

# Works (with -999 masking)
def tagNetworkNew(trackShape):

    trackInput = keras.layers.Input(trackShape, name = 'trackInput')

    tracks = keras.layers.Masking(mask_value = -999, name = 'maskFeatures')(trackInput)

    tracks = keras.layers.TimeDistributed(keras.layers.Dense(8, activation = 'relu'), name = 'td_dense1')(tracks)

    fourVecInput = keras.layers.Input((100, 4), name = 'fourVecInput')

    attention_outputs = keras.layers.GRU(8, activation = 'relu', return_sequences = False, return_state = False, name = 'noseq_gru')(tracks)

    tracks = keras.layers.Dense(8, activation = 'relu', name = 'out_dense_1')(attention_outputs)

    outputTag = keras.layers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return keras.Model(inputs = [trackInput, fourVecInput], outputs = outputTag)

# Works (with -999 masking, 0 less well)
def tagNetworkNewAttn(trackShape):

    trackInput = keras.layers.Input(trackShape, name = 'trackInput')

    tracks = keras.layers.Masking(mask_value = 0, name = 'maskFeatures')(trackInput)

    tracks = keras.layers.TimeDistributed(keras.layers.Dense(8, activation = 'relu'), name = 'td_dense1')(tracks)

    fourVecInput = keras.layers.Input((100, 4), name = 'fourVecInput')

    tracks, hidden = keras.layers.GRU(8, activation = 'relu', return_sequences = True, return_state = True, name = 'noseq_gru')(tracks)

    tracks, attention_weights = Attention(8)([hidden, tracks])

    tracks = keras.layers.Dense(8, activation = 'relu', name = 'out_dense_1')(tracks)

    outputTag = keras.layers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return keras.Model(inputs = [trackInput, fourVecInput], outputs = outputTag)

# Use attention_weights
def tagNetworkNewAttnW(trackShape):

    trackInput = keras.layers.Input(trackShape, name = 'trackInput')

    tracks = keras.layers.Masking(mask_value = 0, name = 'maskFeatures')(trackInput)

    tracks = keras.layers.TimeDistributed(keras.layers.Dense(8, activation = 'relu'), name = 'td_dense1')(tracks)

    fourVecInput = keras.layers.Input((100, 4), name = 'fourVecInput')

    tracks, hidden = keras.layers.GRU(8, activation = 'relu', return_sequences = True, return_state = True, name = 'noseq_gru')(tracks)

    tracks, attention_weights = Attention(8)([hidden, tracks])

    w = keras.layers.GRU(8, activation = 'relu', return_sequences = False, return_state = False, name = 'a_gru')(attention_weights)

    tracks = keras.layers.Concatenate()([tracks, w])

    tracks = keras.layers.Dense(8, activation = 'relu', name = 'out_dense_1')(tracks)

    outputTag = keras.layers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return keras.Model(inputs = [trackInput, fourVecInput], outputs = outputTag)

# Working - bit slow to get going
def tagNetworkNewAttn4v(trackShape):

    trackInput = keras.layers.Input(trackShape, name = 'trackInput')

    tracks = keras.layers.Masking(mask_value = 0, name = 'maskFeatures')(trackInput)

    tracks = keras.layers.TimeDistributed(keras.layers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)

    fourVecInput = keras.layers.Input((100, 4), name = 'fourVecInput')
    fourVectors = keras.layers.BatchNormalization(axis = 1)(fourVecInput)

    fourVectors = keras.layers.TimeDistributed(keras.layers.Dense(2, activation = 'relu'), name = 'td_dense2')(fourVectors)

    tracks = keras.layers.Concatenate()([tracks, fourVectors])

    tracks, hidden = keras.layers.GRU(8, activation = 'relu', return_sequences = True, return_state = True, name = 'noseq_gru')(tracks)

    tracks, attention_weights = Attention(8)([hidden, tracks])

    tracks = keras.layers.Dense(8, activation = 'relu', name = 'out_dense_1')(tracks)

    outputTag = keras.layers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return keras.Model(inputs = [trackInput, fourVecInput], outputs = outputTag)

# With more 4Vec TD layers - works!
def tagNetworkNewAttn4v2(trackShape):

    trackInput = keras.layers.Input(trackShape, name = 'trackInput')

    tracks = keras.layers.Masking(mask_value = 0, name = 'maskFeatures')(trackInput)

    tracks = keras.layers.TimeDistributed(keras.layers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)

    fourVecInput = keras.layers.Input((100, 4), name = 'fourVecInput')
    fourVectors = keras.layers.BatchNormalization(axis = 1)(fourVecInput)

    fourVectors = keras.layers.TimeDistributed(keras.layers.Dense(8, activation = 'relu'), name = 'td_dense2')(fourVectors)

    tracks = keras.layers.Concatenate()([tracks, fourVectors])

    tracks, hidden = keras.layers.GRU(8, activation = 'relu', return_sequences = True, return_state = True, name = 'noseq_gru')(tracks)

    tracks, attention_weights = Attention(8)([hidden, tracks])

    tracks = keras.layers.Dense(8, activation = 'relu', name = 'out_dense_1')(tracks)

    outputTag = keras.layers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return keras.Model(inputs = [trackInput, fourVecInput], outputs = outputTag)

# GRU rather than TD - works okay
def tagNetworkNewAttn4v3(trackShape):

    trackInput = keras.layers.Input(trackShape, name = 'trackInput')

    tracks = keras.layers.Masking(mask_value = 0, name = 'maskFeatures')(trackInput)

    tracks = keras.layers.TimeDistributed(keras.layers.Dense(32, activation = 'relu'), name = 'td_dense1')(tracks)

    fourVecInput = keras.layers.Input((100, 4), name = 'fourVecInput')
    fourVectors = keras.layers.BatchNormalization(axis = 1)(fourVecInput)

    fourVectors = keras.layers.GRU(8, activation = 'relu', return_sequences = True, return_state = False, name = 'noseq_gruV')(fourVectors)

    tracks = keras.layers.Concatenate()([tracks, fourVectors])

    tracks, hidden = keras.layers.GRU(8, activation = 'relu', return_sequences = True, return_state = True, name = 'noseq_gru')(tracks)

    tracks, attention_weights = Attention(8)([hidden, tracks])

    tracks = keras.layers.Dense(8, activation = 'relu', name = 'out_dense_1')(tracks)

    outputTag = keras.layers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return keras.Model(inputs = [trackInput, fourVecInput], outputs = outputTag)

# tensorflow.python.framework.errors_impl.InvalidArgumentError: Nan in summary histogram for: batch_normalization_1/beta_0
#          [[{{node batch_normalization_1/beta_0}}]]
#          [[{{node soft_reco_layer_1/BroadcastTo}}]]
# tensorflow.python.framework.errors_impl.InvalidArgumentError: Nan in summary histogram for: td_dense1/bias_0
#          [[{{node td_dense1/bias_0}}]]
#          [[{{node noseq_gru1/TensorArrayUnstack_1/Shape}}]]

# With or without masking

# Overtrains (maskVal = -999, forCNN = True)
def tagNetworkNewSoftReco(trackShape):

    reg = {'kernel_regularizer' : keras.regularizers.l2(0.)}

    trackInput = keras.layers.Input(trackShape, name = 'trackInput')

    tracks = keras.layers.Masking(mask_value = 0, name = 'maskFeatures')(trackInput)

    tracks = keras.layers.TimeDistributed(keras.layers.Dense(32, activation = 'relu', **reg), name = 'td_dense1')(tracks)

    tracks, hidden = keras.layers.GRU(8, activation = 'relu', return_sequences = True, return_state = True, name = 'noseq_gru1', **reg)(tracks)

    fourVecInput = keras.layers.Input((100, 4), name = 'fourVecInput')

    fourVectors = keras.layers.Masking(mask_value = 0, name = 'mask4V')(fourVecInput)

    _, _, fourVectors = SoftRecoLayer(8)([hidden, tracks, fourVectors])

    fourVectors = keras.layers.BatchNormalization(axis = -1)(fourVectors)

    fourVectors = keras.layers.GRU(8, activation = 'relu', return_sequences = False, **reg)(fourVectors)

    tracks = keras.layers.GRU(8, activation = 'relu', return_sequences = False, return_state = False, name = 'noseq_gru2', **reg)(tracks)

    tracks = keras.layers.Concatenate()([tracks, fourVectors])

    tracks = keras.layers.Dense(8, activation = 'relu', name = 'out_dense_1', **reg)(tracks)

    outputTag = keras.layers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return keras.Model(inputs = [trackInput, fourVecInput], outputs = outputTag)

# Works! But does it weight correctly? No Reg
def tagNetworkNewSoftReco2(trackShape):

    reg = {'kernel_regularizer' : keras.regularizers.l2(0.)}

    trackInput = keras.layers.Input(trackShape, name = 'trackInput')

    tracks = keras.layers.Masking(mask_value = 0, name = 'maskFeatures')(trackInput)

    tracks = keras.layers.TimeDistributed(keras.layers.Dense(32, activation = 'relu', **reg), name = 'td_dense1')(tracks)

    tracks, hidden = keras.layers.GRU(8, activation = 'relu', return_sequences = True, return_state = True, name = 'noseq_gru1', **reg)(tracks)

    fourVecInput = keras.layers.Input((100, 4), name = 'fourVecInput')

    fourVectors = keras.layers.Masking(mask_value = 0, name = 'mask4V')(fourVecInput)

    _, _, fourVectors = SoftRecoLayer(8)([hidden, tracks, fourVectors])

    fourVectors = keras.layers.BatchNormalization(axis = -1)(fourVectors)

    fourVectors = keras.layers.GRU(8, activation = 'relu', return_sequences = False, **reg)(fourVectors)

    tracks = keras.layers.GRU(8, activation = 'relu', return_sequences = False, return_state = False, name = 'noseq_gru2', **reg)(tracks)

    tracks = keras.layers.Concatenate()([tracks, fourVectors])

    tracks = keras.layers.Dense(8, activation = 'relu', name = 'out_dense_1', **reg)(tracks)

    outputTag = keras.layers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return keras.Model(inputs = [trackInput, fourVecInput], outputs = outputTag)

# Try with a different config to use 4v
def tagNetworkNewSoftReco3(trackShape):

    reg = {'kernel_regularizer' : keras.regularizers.l2(0.)}

    trackInput = keras.layers.Input(trackShape, name = 'trackInput')

    tracks = keras.layers.Masking(mask_value = 0, name = 'maskFeatures')(trackInput)

    tracks = keras.layers.TimeDistributed(keras.layers.Dense(8, activation = 'swish', **reg), name = 'td_dense1')(tracks)

    tracks4v, hidden = keras.layers.GRU(8, activation = 'swish', return_sequences = True, return_state = True, name = 'noseq_gru1', **reg)(tracks)

    fourVecInput = keras.layers.Input((100, 4), name = 'fourVecInput')

    fourVectors = keras.layers.Masking(mask_value = 0, name = 'mask4V')(fourVecInput)

    _, _, fourVectors = SoftRecoLayer(8)([hidden, tracks4v, fourVectors])

    fourVectors = keras.layers.BatchNormalization(axis = 1, momentum = 0.1)(fourVectors)

    fourVectors = keras.layers.TimeDistributed(keras.layers.Dense(8, activation = 'swish', **reg), name = 'td_dense2')(fourVectors)

    tracks = keras.layers.Concatenate()([tracks, fourVectors])

    tracks = keras.layers.GRU(8, activation = 'swish', return_sequences = False, return_state = False, name = 'noseq_gru2', **reg)(tracks)

    # tracks = keras.layers.Dense(8, activation = 'swish', name = 'out_dense_1', **reg)(tracks)

    outputTag = keras.layers.Dense(1, activation='sigmoid', name = 'outputTag')(tracks)

    return keras.Model(inputs = [trackInput, fourVecInput], outputs = outputTag)

#Training configuration
maxtracks      = 100
batch_size     = 2 ** 12

TRACK_SHAPE = (100, 18) # nTracks, nFeatures

nTrackCategories = 4

# model = tagNetworkSoftReco(TRACK_SHAPE)
# model = tagNetworkNewAttnW(TRACK_SHAPE)
model = tagNetworkNewSoftReco3(TRACK_SHAPE)
model.summary(line_length = 200)

pprint(model.get_config())

# adam = Adam(lr = 1E-2, amsgrad = True, clipvalue = 5.0)
adam = keras.optimizers.Nadam(lr = 1E-3, clipvalue = 5.0)
# adam = SGD(lr = 1E-3)
earlyStopping = EarlyStopping(patience = 100)

model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['accuracy'])

generatorOptions = {

'trainingType' : 'tag_plus_4v',

'featureName' : 'featureArray',

'catName' : 'catArray',

'tagName' : 'tagArray',

'extrasFileName' : '/home/dan/ftData/DTT_MC2015_Reco15aStrip24_DIMUON_Bd2JpsiKstar.h5',

'extrasFeatureName' : 'fourVectorArray',

'forCNN' : True, # Set -999 to zero
'maskVal' : -999,

'useExtraFeatures' : True,

'useWeights' : True,

'nClasses' : 4,
'nFeatures' : 18,


'trainFrac' : 0.8,
'validationFrac' : 0.1,
'testFrac' : 0.1,

'batchSize' : 2 ** 14,

# 'dataSize' : 100000

}

# tb = TensorBoard(log_dir='logsTest/softReco8', histogram_freq = 1, write_grads = True, batch_size = 2 ** 12)

# inputFiles = ['/Users/MBP/GoogleDrive/ftag/DTT_MC2016_Reco16Strip26_Down_DIMUON_Bu2JpsiK.h5',
#               '/Users/MBP/GoogleDrive/ftag/DTT_MC2016_Reco16Strip26_Down_DIMUON_Bu2JpsiK_copy.h5']

inputFiles = '/home/dan/ftData/DTT_MC2015_Reco15aStrip24_DIMUON_Bd2JpsiKstar.h5'

# dataGeneratorTrain = DataGenerator(inputFiles, dataset = 'train', **generatorOptions)
#
# # Probably don't need a generator for these, but we might as well
# # Although validation slows down training, so if this is small perhaps we can hold it in RAM
# dataGeneratorValidation = DataGenerator(inputFiles, dataset = 'validation', **generatorOptions)
# dataGeneratorTest = DataGenerator(inputFiles, dataset = 'test', **generatorOptions)

genTrain, genValidation, genTest = createSplitGenerators(inputFiles,
                                                         generatorOptions,
                                                         shuffle = False,
                                                         shuffleChunks = False)

model.fit_generator(generator = genTrain,
                    validation_data=[genValidation._data_generation_features_4v(list(genValidation.indices)[:10000])[0],
                                     genValidation._data_generation_tag(list(genValidation.indices)[:10000])[0]],
                    # callbacks = [earlyStopping, tb, TerminateOnNaN()],
                    epochs = 500, verbose = 1)

# Get the tags for the full training sample, so that these can be used to calculate the ROC
y_train = genTrain.getTags()
y_test = genTest.getTags()

# Can use the generators for prediction too, but need to ensure that there is no shuffling wrt the above
y_out_train = model.predict_generator(genTrain)
y_out_test = model.predict_generator(genTest)

rocAUC_train = roc_auc_score(y_train, y_out_train)
rocAUC_test = roc_auc_score(y_test, y_out_test)

print('ROC Train:', rocAUC_train)
print('ROC Test:', rocAUC_test)

modelName = 'TestModel'

makeTrainingPlots(model)
saveModel(model, modelName)
exportForCalibration(y_test, y_out_test)
