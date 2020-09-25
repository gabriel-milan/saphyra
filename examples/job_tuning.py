#!/usr/bin/env python

try:
  from tensorflow.compat.v1 import ConfigProto
  from tensorflow.compat.v1 import InteractiveSession

  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)
except Exception as e:
  print(e)
  print("Not possible to set gpu allow growth")



def getPatterns( path, cv, sort):

  from Gaugi import load

  def norm1( data ):
      norms = np.abs( data.sum(axis=1) )
      norms[norms==0] = 1
      #return np.expand_dims( data/norms[:,None], axis=2 )
      return data/norms[:,None]

  # Load data
  d = load(path)
  feature_names = d['features'].tolist()

  # Get the normalized rings
  data_rings = norm1(d['data'][:,1:101])
  
  # How many events?
  n = data_rings.shape[0]

  # extract all shower shapes
  data_reta   = d['data'][:, feature_names.index('reta')].reshape((n,1))
  data_rphi   = d['data'][:, feature_names.index('rphi')].reshape((n,1))
  data_eratio = d['data'][:, feature_names.index('eratio')].reshape((n,1))
  data_weta2  = d['data'][:, feature_names.index('weta2')].reshape((n,1))
  data_f1     = d['data'][:, feature_names.index('f1')].reshape((n,1))
  
  # Get the mu average 
  data_mu     = d['data'][:, feature_names.index('avgmu')].reshape((n,1))
  target = d['target']

  # This is mandatory
  splits = [(train_index, val_index) for train_index, val_index in cv.split(data_mu,target)]
  
  data_shower_shapes = np.concatenate( (data_reta,data_rphi,data_eratio,data_weta2,data_f1), axis=1)

    # split for this sort
  x_train = [ data_rings[splits[sort][0]],   data_shower_shapes [ splits[sort][0] ] ]
  x_val   = [ data_rings[splits[sort][1]],   data_shower_shapes [ splits[sort][1] ] ]
  y_train = target [ splits[sort][0] ]
  y_val   = target [ splits[sort][1] ]

  return x_train, x_val, y_train, y_val, splits




def getPileup( path ):
  from Gaugi import load
  return load(path)['data'][:,0]


def getJobConfigId( path ):
  from Gaugi import load
  return dict(load(path))['id']



import numpy as np
import argparse
import sys,os


parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()


parser.add_argument('-o','--outputFile', action='store',
        dest='outputFile', required = False, default = None,
            help = "The output tuning name.")

parser.add_argument('-d','--dataFile', action='store',
        dest='dataFile', required = False, default = None,
            help = "The data/target file used to train the model.")

parser.add_argument('-r','--refFile', action='store',
        dest='refFile', required = False, default = None,
            help = "The reference file.")

if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args()


outputFile = args.outputFile





#
# Build my model
#

import tensorflow as tf
from tensorflow.keras import layers
from saphyra.layers import RpLayer, rvec
# Ringer NN
input_rings = layers.Input(shape=(100,), name='Input_rings')
#input_rings  = RpLayer(rvec, name='RpLayer')(input_rings)
conv_rings   = layers.Reshape((100,1))(input_rings)
conv_rings   = layers.Conv1D( 16, kernel_size=2, name='conv_1', activation='relu')(conv_rings)
conv_rings   = layers.Conv1D( 32, kernel_size=2, name='conv_2', activation='relu')(conv_rings)
conv_output  = layers.Flatten()(conv_rings)
# Shower shape NN
input_shower_shapes = layers.Input(shape=(5,), name='Input_shower_shapes')
dense_shower_shapes = layers.Dense(4, activation='relu', name='dense_shower_shapes_1')(input_shower_shapes)
# Decision NN
input_concat = layers.Concatenate(axis=1)([conv_output, dense_shower_shapes])
dense = layers.Dense(32, activation='relu', name='dense_layer')(input_concat)
dense = layers.Dense(1,activation='linear', name='output_for_inference')(dense)
output = layers.Activation('sigmoid', name='output_for_training')(dense)
# Build the model
model = tf.keras.Model([input_rings, input_shower_shapes], output, name = "model")
model.summary()


#
# reference configuration
#

ref_target = [
              ('tight_cutbased' , 'T0HLTElectronT2CaloTight'        ),
              ('medium_cutbased', 'T0HLTElectronT2CaloMedium'       ),
              ('loose_cutbased' , 'T0HLTElectronT2CaloLoose'        ),
              ('vloose_cutbased', 'T0HLTElectronT2CaloVLoose'       ),
              ]




from saphyra.decorators import Summary, Reference
decorators = [Summary(), Reference(args.refFile, ref_target) ]

from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='val_sp', mode='max', verbose=1, patience=25, restore_best_weights=True)



import datetime, os
from tensorflow.keras.callbacks import TensorBoard
logdir = os.path.join('.', 'logs/%s' %(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
tensorboard = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


from saphyra import PatternGenerator
from saphyra.metrics import sp_metric, pd_metric, fa_metric
from sklearn.model_selection import StratifiedKFold
from saphyra.applications import BinaryClassificationJob


job = BinaryClassificationJob(  PatternGenerator( args.dataFile, getPatterns ),
                                StratifiedKFold(n_splits=10, random_state=512, shuffle=True),
                                loss              = 'binary_crossentropy',
                                metrics           = ['accuracy', sp_metric, pd_metric, fa_metric],
                                callbacks         = [stop, tensorboard],
                                epochs            = 2,
                                class_weight      = True,
                                sorts             = 1,
                                inits             = 1,
                                models            = [model],
                                )

job.decorators += decorators

# Run it!
job.run()








