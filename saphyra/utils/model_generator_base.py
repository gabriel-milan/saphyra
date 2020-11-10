
__init__ = ["model_generator_base"]

import tensorflow as tf
from tensorflow.keras.models import clone_model, Model
from tensorflow.keras import layers
from saphyra.core import TunedDataReader
from Gaugi.messenger.macros import *
from Gaugi import Logger




class model_generator_base( Logger ):

  #
  # Constructor
  #
  def __init__( self ):
    Logger.__init__(self)


  #
  # Call method
  #
  def __call__( self, sort ):
    pass


  #
  # tranfer the source weights to the target layer
  #
  def transfer_weights( self, from_model, from_layer, to_model, to_layer, trainable=True ):

    weights = None
    # Loop over all layers in the source model
    for layer in from_model.layers:
      if layer.name == from_layer:
        weights = layer.weights
    
    if not weights:
      MSG_FATAL( self, "From model with layer %s does not exist.", from_layer)

    # Loop of all layers in the target model
    for layer in to_model.layers:
      if layer.name == to_layer:
        if layer.weights.shape != weighhs.shape:
          MSG_FATAL(self, "The target layer with name %s does not match with the weights shape" , to_layer )
        layer.weights = weights
        layer.trainable = trainable



  #
  # Load all tuned filed
  #
  def load_models( self, path ):
    tunedData = TunedDataReader()
    tunedData.load( path )
    return tunedData.obj().get_data()



  #
  # Get best model given the sort number
  #
  def get_best_model( self , tuned_list, sort, imodel):

    best_model=None; best_sp=-999
    # Loop over all tuned files
    for tuned in tuned_list:
      history = tuned['history']
      if tuned['sort']==sort and best_sp > history['summary']['max_sp_op'] and tuned['imodel']==imodel: 
        # read the model 
        best_model = model_from_json( json.dumps(tuned['sequence'], separators=(',', ':')) )
        best_model.set_weights( tuned['weights'] )
        best_model = Model(model.inputs, model.layers[-1].output)
        best_sp =history['summary']['max_sp_op']

    return best_model
    





 
