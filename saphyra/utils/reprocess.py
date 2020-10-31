

__all__ = ["reprocess"]

from Gaugi import Logger, StatusCode, expandFolders, mkdir_p, load, save
from Gaugi.messenger.macros import *

from pprint import pprint

from saphyra.layers.RpLayer import RpLayer
from saphyra.core.readers.versions import TunedData_v1
from saphyra.core import Context


# Just to remove the keras dependence
import tensorflow as tf
model_from_json = tf.keras.models.model_from_json
import json


class Reprocess( Logger ):

  def __init__(self):

    Logger.__init__(self)
    self.__context = Context()



  #
  # run job
  #
  def __call__( self , generator, files, outputfile, crossval, decorators):


    files =  expandFolders(files) 

    mkdir_p( outputfile )

    pprint(files)

    for idx, file in enumerate(files):
    
      MSG_INFO( self, "Opening file %s...", file )
      raw = load(file)

      tunedData = TunedData_v1()

      for jdx, tuned in enumerate(raw['tunedData']):

        # force the context is empty for each iteration
        self.__context.clear()


        sort = tuned['sort']
        init = tuned['init']
        imodel = tuned['imodel']
        history = tuned['history']

        # get the current kfold and train, val sets
        x_train, x_val, y_train, y_val, index_from_cv = self.pattern_g( generator, crossval, sort )

        # recover keras model
        model = model_from_json( json.dumps(tuned['sequence'], separators=(',', ':')) )#, custom_objects={'RpLayer':RpLayer} )
        model.set_weights( tuned['weights'] )


        # Should not be store
        self.__context.setHandler( "valData" , (x_val, y_val)       )
        self.__context.setHandler( "trnData" , (x_train, y_train)   )
        self.__context.setHandler( "index"   , index_from_cv        )
        self.__context.setHandler( "crossval", crossval             )


        # It will be store into the file
        self.__context.setHandler( "model"   , model         )
        self.__context.setHandler( "sort"    , sort          )
        self.__context.setHandler( "init"    , init          )
        self.__context.setHandler( "imodel"  , imodel        )
        self.__context.setHandler( "time"    , tuned['time'] )
        self.__context.setHandler( "history" , history       )


        for tool in decorators:
          #MSG_INFO( self, "Executing the pos processor %s", tool.name() )
          tool.decorate( history, self.__context )

        tunedData.attach_ctx( self.__context )


      try:
        MSG_INFO( self, "Saving file..." )
        tunedData.save( outputfile+'/'+ file.split('/')[-1] )
      except Exception as e:
        MSG_FATAL( self, "Its not possible to save the tuned data: %s" , e )


    return StatusCode.SUCCESS




  def pattern_g( self, generator, crossval, sort ):
    # If the index is not set, you muat run the cross validation Kfold to get the index
    # this generator must be implemented by the user
    return generator(crossval, sort)





reprocess = Reprocess()

