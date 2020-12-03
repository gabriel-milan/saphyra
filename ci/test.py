

__all__ = ['crossval_table', 'get_color_fader']


from Gaugi.tex import *
from Gaugi.messenger.macros import *
from Gaugi import Logger, expandFolders, load
from functools import reduce
import collections, os, glob, json, copy, re
import numpy as np
import pandas as pd
import os

from saphyra.layers.RpLayer import RpLayer
# Just to remove the keras dependence
import tensorflow as tf
model_from_json = tf.keras.models.model_from_json
import json

try:
  import matplotlib.pyplot as plt
  import matplotlib as mpl
  #import seaborn as sns
  #import mplhep as hep
  #plt.style.use(hep.style.ATLAS)
except:
  print('no matplotlib found!')





from saphyra import crossval_table
    
#
# My local test to debug the cross validation table class
#

def create_op_dict(op):
    d = {
              op+'_pd_ref'    : "reference/"+op+"_cutbased/pd_ref#0",
              op+'_fa_ref'    : "reference/"+op+"_cutbased/fa_ref#0",
              op+'_sp_ref'    : "reference/"+op+"_cutbased/sp_ref",
              op+'_pd_val'    : "reference/"+op+"_cutbased/pd_val#0",
              op+'_fa_val'    : "reference/"+op+"_cutbased/fa_val#0",
              op+'_sp_val'    : "reference/"+op+"_cutbased/sp_val",
              op+'_pd_op'     : "reference/"+op+"_cutbased/pd_op#0",
              op+'_fa_op'     : "reference/"+op+"_cutbased/fa_op#0",
              op+'_sp_op'     : "reference/"+op+"_cutbased/sp_op",
            
              # Counts
              op+'_pd_ref_passed'    : "reference/"+op+"_cutbased/pd_ref#1",
              op+'_fa_ref_passed'    : "reference/"+op+"_cutbased/fa_ref#1",
              op+'_pd_ref_total'     : "reference/"+op+"_cutbased/pd_ref#2",
              op+'_fa_ref_total'     : "reference/"+op+"_cutbased/fa_ref#2",   
              op+'_pd_val_passed'    : "reference/"+op+"_cutbased/pd_val#1",
              op+'_fa_val_passed'    : "reference/"+op+"_cutbased/fa_val#1",
              op+'_pd_val_total'     : "reference/"+op+"_cutbased/pd_val#2",
              op+'_fa_val_total'     : "reference/"+op+"_cutbased/fa_val#2",  
              op+'_pd_op_passed'     : "reference/"+op+"_cutbased/pd_op#1",
              op+'_fa_op_passed'     : "reference/"+op+"_cutbased/fa_op#1",
              op+'_pd_op_total'      : "reference/"+op+"_cutbased/pd_op#2",
              op+'_fa_op_total'      : "reference/"+op+"_cutbased/fa_op#2",
    } 
    return d


tuned_info = collections.OrderedDict( {
              # validation
              "max_sp_val"      : 'summary/max_sp_val',
              "max_sp_pd_val"   : 'summary/max_sp_pd_val#0',
              "max_sp_fa_val"   : 'summary/max_sp_fa_val#0',
              # Operation
              "max_sp_op"       : 'summary/max_sp_op',
              "max_sp_pd_op"    : 'summary/max_sp_pd_op#0',
              "max_sp_fa_op"    : 'summary/max_sp_fa_op#0',
              #"loss"            : 'loss',
              #"val_loss"        : 'val_loss',
              #"accuracy"        : 'accuracy',
              #"val_accuracy"    : 'val_accuracy',
              #"max_sp_best_epoch_val": 'max_sp_best_epoch_val',
              } )

tuned_info.update(create_op_dict('tight'))
tuned_info.update(create_op_dict('medium'))
tuned_info.update(create_op_dict('loose'))
tuned_info.update(create_op_dict('vloose'))


#etbins = [15,20,30,40,50,100000]
etbins = [15,20]
etabins = [0, 0.8 , 1.37, 1.54, 2.37, 2.5]

cv  = crossval_table( tuned_info, etbins = etbins , etabins = etabins )

cv.fill( 'tt/*.et0*/*/*.pic.gz', 'v10')
#cv.to_csv( 'v11.csv' )

best_inits = cv.filter_inits("max_sp_val")
best_inits = best_inits.loc[(best_inits.model_idx==0)]
best_sorts = cv.filter_sorts(best_inits, 'max_sp_op')


#cv.plot_training_curves( best_inits, best_sorts, 'v10' )

for op in ['tight','medium','loose','vloose']:
   cv.dump_beamer_table( best_inits ,  [op],
                    'tuning_'+op,
                    title = op+' Tunings',
                    tags = ['v10'],
                    )


