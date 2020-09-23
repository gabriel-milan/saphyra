
__all__ = [
  "auc",
  "f1_score",
  "SP_Metric",
]

from tensorflow.keras.metrics import AUC
from tensorflow.keras import backend as K


def auc(y_true, y_pred, num_thresholds=2000):
  import tensorflow as tf
  auc = tf.metrics.auc(y_true, y_pred,num_thresholds=num_thresholds)[1]
  K.get_session().run(tf.local_variables_initializer())
  return auc


def f1_score(y_true, y_pred):
  import tensorflow as tf
  f1 = tf.contrib.metrics.f1_score(y_true, y_pred)[1]
  K.get_session().run(tf.local_variables_initializer())
  return f1

###
#
# => SP, PD, FA metric
# 
# PS: name is SP_Metric so it won't
# conflict with legacy "sp" metric,
# which is actually a callback.
#
# Using this class, you may add it to
# your `model.compile` method like the
# following:
#
# model.compile (optimizer='...', loss='...', metrics=[SP_Metric()]) 
#
###
class SP_Metric (AUC):

  # This implementation works with Tensorflow backend tensors.
  # That way, calculations happen faster and results can be seen
  # while training, not only after each epoch

  def result(self):

    # Add K.epsilon() for forbiding division by zero
    fa = self.false_positives / (self.true_negatives + self.false_positives + K.epsilon())
    pd = self.true_positives  / (self.true_positives + self.false_positives + K.epsilon())
    sp = K.sqrt(  K.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )
    knee = K.argmax(sp)
    return sp[knee]

class PD_Metric (AUC):

  # This implementation works with Tensorflow backend tensors.
  # That way, calculations happen faster and results can be seen
  # while training, not only after each epoch

  def result(self):

    # Add K.epsilon() for forbiding division by zero
    fa = self.false_positives / (self.true_negatives + self.false_positives + K.epsilon())
    pd = self.true_positives  / (self.true_positives + self.false_positives + K.epsilon())
    sp = K.sqrt(  K.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )
    knee = K.argmax(sp)
    return pd[knee]

class FA_Metric (AUC):

  # This implementation works with Tensorflow backend tensors.
  # That way, calculations happen faster and results can be seen
  # while training, not only after each epoch

  def result(self):

    # Add K.epsilon() for forbiding division by zero
    fa = self.false_positives / (self.true_negatives + self.false_positives + K.epsilon())
    pd = self.true_positives  / (self.true_positives + self.false_positives + K.epsilon())
    sp = K.sqrt(  K.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )
    knee = K.argmax(sp)
    return fa[knee]