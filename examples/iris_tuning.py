#!/usr/bin/env python
import os

def getPatterns( path, cv, sort):

  from sklearn.datasets import load_iris
  iris = load_iris()
  data = iris['data'][:100]
  target = iris['target'][:100]
  # This is mandatory
  splits = [(train_index, val_index) for train_index, val_index in cv.split(data,target)]
  # split for this sort
  x_train = data[splits[sort][0]]
  x_val   = data[splits[sort][1]]
  y_train = target [ splits[sort][0] ]
  y_val   = target [ splits[sort][1] ]
  return x_train, x_val, y_train, y_val, splits




import tensorflow as tf
from tensorflow.keras import layers
input = layers.Input(shape=(4,), name='Input')
dense = layers.Dense(2, activation='tanh', name='dense1')(input)
dense = layers.Dense(1,activation='linear', name='output_for_inference')(dense)
output = layers.Activation('sigmoid', name='output_for_training')(dense)
model = tf.keras.Model(input, output, name = "model")


model.get_layer('dense1').trainable=False


from saphyra.decorators import Summary
decorators = [Summary()]

from saphyra.metrics import sp_metric, pd_metric, fa_metric
from tensorflow.keras.callbacks import EarlyStopping
stop = EarlyStopping(monitor='val_sp', mode='max', verbose=1, patience=25, restore_best_weights=True)


from tensorflow.keras.callbacks import TensorBoard
import datetime
#logdir = os.path.join('.', 'logs/%s' %(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
#tensorboard = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


from saphyra import PatternGenerator
from sklearn.model_selection import StratifiedKFold
from saphyra.applications import BinaryClassificationJob

job = BinaryClassificationJob(  PatternGenerator( "", getPatterns ),
                                StratifiedKFold(n_splits=10, random_state=512, shuffle=True),
                                loss              = 'binary_crossentropy',
                                metrics           = ['accuracy', sp_metric, pd_metric, fa_metric],
                                callbacks         = [stop],
                                epochs            = 50,
                                class_weight      = True,
                                sorts             = [0],
                                inits             = 1,
                                models            = [model],
                                )

job.decorators += decorators


# Run it!
job.run()






