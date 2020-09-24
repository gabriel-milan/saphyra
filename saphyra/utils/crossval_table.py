

__all__ = ['crossval_table']


from Gaugi.tex import *
from Gaugi.messenger.macros import *
from Gaugi import Logger, expandFolders, load
from functools import reduce
import collections, os, glob, json, copy, re
import numpy as np
import pandas as pd



class crossval_table( Logger ):

    def __init__(self, config_dict): 
        
        Logger.__init__(self)
        self.__table = None
        # Check wanted key type 
        self.__config_dict = collections.OrderedDict(config_dict) if type(config_dict) is dict else config_dict
      


    #
    # Fill the main dataframe with values from the tuning files and convert to pandas dataframe
    #
    def fill(self, path, tag):
        
        paths = expandFolders( path )
        MSG_INFO(self, "Reading file for %s tag from %s", tag , path)

        # Creating the dataframe
        dataframe = collections.OrderedDict({
                              'train_tag'      : [],
                              'et_bin'         : [],
                              'eta_bin'        : [],
                              'model_idx'      : [],
                              'sort'           : [],
                              'init'           : [],
                              'file_name'      : [],
                              'tuned_idx'      : [],
                          })

        # Complete the dataframe for each varname in the config dict
        for varname in self.__config_dict.keys():
            dataframe[varname] = []

        MSG_INFO(self, 'There are %i files for this task...' %(len(paths)))
        MSG_INFO(self, 'Filling the table... ')

        for ituned_file_name in paths:
            gfile = load(ituned_file_name)
            tuned_file = gfile['tunedData']
            
            for idx, ituned in enumerate(tuned_file):
                history = ituned['history']
                # get the basic from model
                dataframe['train_tag'].append(tag)
                dataframe['model_idx'].append(ituned['imodel'])
                dataframe['sort'].append(ituned['sort'])
                dataframe['init'].append(ituned['init'])
                dataframe['et_bin'].append(self.get_etbin(ituned_file_name))
                dataframe['eta_bin'].append(self.get_etabin(ituned_file_name))
                dataframe['file_name'].append(ituned_file_name)
                dataframe['tuned_idx'].append( idx )
                # Get the value for each wanted key passed by the user in the contructor args.
                for key, local  in self.__config_dict.items():
                    dataframe[key].append( self.__get_value( history, local ) )
        
        self.__table = self.__table.append( pd.DataFrame(dataframe) ) if not self.__table is None else pd.DataFrame(dataframe)
        MSG_INFO(self, 'End of fill step, a pandas DataFrame was created...')


    #
    # Get the value using recursive dictionary navigation
    #
    def __get_value(self, history, local):
        
        # Protection to not override the history since this is a 'mutable' object
        var = copy.copy(history)
        for key in local.split('/'):
            var = var[key.split('#')[0]][int(key.split('#')[1])] if '#' in key else var[key]
        return var



    def get_etbin(self, job):
        return int(  re.findall(r'et[a]?[0-9]', job)[0][-1] )



    def get_etabin(self, job):
        return int( re.findall(r'et[a]?[0-9]',job)[1] [-1] )



    #
    # Get the pandas dataframe
    #
    def table(self):
        return self.__table



    #
    # Return only best inits
    #
    def filter_inits(self, key):
        return self.table().loc[self.table().groupby(['et_bin', 'eta_bin', 'model_idx', 'sort'])[key].idxmax(), :]
   

    #
    # Get the best sorts from best inits table
    #
    def filter_sorts(self, best_inits, key):
        return best_inits.loc[best_inits.groupby(['et_bin', 'eta_bin', 'model_idx'])[key].idxmax(), :]


    #
    # Calculate the mean/std table from best inits table
    #
    def describe(self, best_inits ):
    
        # Create a new dataframe to hold this table
        dataframe = { 'train_tag' : [], 'et_bin' : [], 'eta_bin' : []}
        # Include all wanted keys into the dataframe
        for key in self.__config_dict.keys():
            if 'passed' in key: # Skip counts
                continue
            elif ('op' in key) or ('val' in key):
                dataframe[key+'_mean'] = []; dataframe[key+'_std'] = [];
            else:
                dataframe[key] = []
        
        # Loop over all tuning tags and et/eta bins
        for tag in best_inits.train_tag.unique():
            for et_bin in best_inits.et_bin.unique():
                for eta_bin in best_inits.eta_bin.unique():

                  cv_bin = best_inits.loc[ (best_inits.train_tag == tag) & (best_inits.et_bin == et_bin) & (best_inits.eta_bin == eta_bin) ]
                  dataframe['train_tag'].append( tag ); dataframe['et_bin'].append( et_bin ); dataframe['eta_bin'].append( eta_bin )
                  for key in self.__config_dict.keys():
                      if 'passed' in key:
                        continue
                      elif ('op' in key) or ('val' in key):
                          dataframe[key+'_mean'].append( cv_bin[key].mean() ); dataframe[key+'_std'].append( cv_bin[key].std() )
                      else:
                          dataframe[key].append( cv_bin[key].unique()[0] )

        # Return the pandas dataframe
        return pd.DataFrame(dataframe)


    #
    # Get tge cross val integrated table from best inits
    #
    def integrate( self, best_inits, tag ):

        keys = [ key for key in self.__config_dict.keys() if 'passed' in key or 'total' in key]
        table = best_inits.loc[best_inits.train_tag==tag].groupby(['sort']).agg(dict(zip( keys, ['sum']*len(keys))))
        for key in keys:
            if 'passed' in key:
                orig_key = key.replace('_passed','')
                values = table[key].div( table[orig_key+'_total'] )
                table[orig_key] = values
                table=table.drop([key],axis=1)
                table=table.drop([orig_key+'_total'],axis=1)  
        
        return table.agg(['mean','std'])



    #
    # Dump all history for each line in the table
    #
    def dump_all_history( self, table, output_path , tag):
        if not os.path.exists( output_path ):
          os.mkdir( output_path )
        for _ , row in table.iterrows():
            if row.train_tag != tag:
              continue
            # Load history
            history = load( row.file_name )['tunedData'][row.tuned_idx]['history']
            history['loc'] = {'et_bin' : row.et_bin, 'eta_bin' : row.eta_bin, 'sort' : row.sort, 'model_idx' : row.model_idx}
            name = 'history_et_%i_eta_%i_model_%i_sort_%i.json' % (row.et_bin,row.eta_bin,row.model_idx,row.sort)
            with open(os.path.join(output_path, '%s' %name), 'w') as fp:
                #json.dump(transform_serialize(history), fp)
                json.dump(str(history), fp)







    #
    # Create the beamer table file
    #
    def dump_beamer_table( self, best_inits, etbins, etabins, operation_points, output, tags=None, title='' ):
       
        cv_table = self.describe( best_inits )

        # Prepare the config dict using the operation points and some default keys
        config_dict = {}
        for operation_point in operation_points:
            config_dict[operation_point] = {}
            # reference val/op standard keys
            for key in ['pd_val', 'sp_val', 'fa_val', 'pd_op', 'sp_op', 'fa_op']:
                config_dict[operation_point][ key + '_mean' ] = operation_point + '_' + key + '_mean'
                config_dict[operation_point][ key + '_std' ] = operation_point + '_' + key + '_std'
            # reference keys
            config_dict[operation_point][ 'pd_ref' ] = operation_point + '_pd_ref'
            config_dict[operation_point][ 'sp_ref' ] = operation_point + '_sp_ref'
            config_dict[operation_point][ 'fa_ref' ] = operation_point + '_fa_ref'
          
        # Create Latex Et bins
        etbins_str = []; etabins_str = []
        for etBinIdx in range(len(etbins)-1):
            etbin = (etbins[etBinIdx], etbins[etBinIdx+1])
            if etbin[1] > 100 :
                etbins_str.append( r'$E_{T}\text{[GeV]} > %d$' % etbin[0])
            else:
                etbins_str.append(  r'$%d < E_{T} \text{[Gev]}<%d$'%etbin )
      
        # Create Latex eta bins
        for etaBinIdx in range( len(etabins)-1 ):
            etabin = (etabins[etaBinIdx], etabins[etaBinIdx+1])
            etabins_str.append( r'$%.2f<\eta<%.2f$'%etabin )
      
        # Default colors
        colorPD = '\\cellcolor[HTML]{9AFF99}'; colorPF = ''; colorSP = ''

        train_tags = cv_table.train_tag.unique() if not tags else tags
      
        # Dictionary to hold all eff values for each operation point, tag and et/eta binning
        summary = {}
      
        # Create the summary with empty values
        for operation_point in config_dict.keys():
            summary[operation_point] = {}
            for tag in train_tags:
                summary[operation_point][tag] = [ [ {} for __ in range(len(etabins)-1) ] for _ in range(len(etbins)-1)]
                for etBinIdx in range(len(etbins)-1):
                    for etaBinIdx in range(len(etabins)-1):
                        for key in config_dict[operation_point]:
                            summary[operation_point][tag][etBinIdx][etaBinIdx][key] = 0.0
      
        # Fill the summary
        for operation_point in config_dict.keys():
            for tag in train_tags:
                tag_table = cv_table.loc[ cv_table.train_tag == tag]
                for etBinIdx in tag_table.et_bin.unique():
                    for etaBinIdx in tag_table.eta_bin.unique():
                        for key in config_dict[operation_point]:
                            # We have only one line for each tag/et/eta bin
                            summary[operation_point][tag][etBinIdx][etaBinIdx][key] = tag_table.loc[ (tag_table.et_bin==etBinIdx) & 
                                (tag_table.eta_bin==etaBinIdx) ][config_dict[operation_point][key]].values[0]
      

        # Apply beamer
        with BeamerTexReportTemplate1( theme = 'Berlin'
                                 , _toPDF = True
                                 , title = title
                                 , outputFile = output
                                 , font = 'structurebold' ):
        
            for operation_point in summary.keys():
                ### Prepare tables
                tuning_names = ['']; tuning_names.extend( train_tags )
                lines1 = []
                lines1 += [ HLine(_contextManaged = False) ]
                lines1 += [ HLine(_contextManaged = False) ]
                lines1 += [ TableLine( columns = ['','','kinematic region'] + reduce(lambda x,y: x+y,[['',s,''] for s in etbins_str]), _contextManaged = False ) ]
                lines1 += [ HLine(_contextManaged = False) ]
                lines1 += [ TableLine( columns = ['Det. Region','Method','Type'] + reduce(lambda x,y: x+y,[[colorPD+r'$P_{D}[\%]$',colorSP+r'$SP[\%]$',colorPF+r'$P_{F}[\%]$'] \
                                      for _ in etbins_str]), _contextManaged = False ) ]
                lines1 += [ HLine(_contextManaged = False) ]
      
                for etaBinIdx in range(len(etabins) - 1):
                    for idx, tag in enumerate( train_tags ):
                        cv_values=[]; ref_values=[]
                        for etBinIdx in range(len(etbins) - 1):
                            d = summary[operation_point][tag][etBinIdx][etaBinIdx]
                            sp = d['sp_val_mean']*100; sp_std = d['sp_val_std']*100
                            pd = d['pd_val_mean']*100; pd_std = d['pd_val_std']*100
                            fa = d['fa_val_mean']*100; fa_std = d['fa_val_std']*100
                            refsp = d['sp_ref']*100; refpd = d['pd_ref']*100; reffa = d['fa_ref']*100
                            cv_values   += [ colorPD+('%1.2f$\pm$%1.2f')%(pd,pd_std),colorSP+('%1.2f$\pm$%1.2f')%(sp,sp_std),colorPF+('%1.2f$\pm$%1.2f')%(fa,fa_std),    ]
                            ref_values  += [ colorPD+('%1.2f')%(refpd), colorSP+('%1.2f')%(refsp), colorPF+('%1.2f')%(reffa)]
                        ### Make summary table
                        if idx > 0:
                            lines1 += [ TableLine( columns = ['', tuning_names[idx+1], 'Cross Validation'] + cv_values   , _contextManaged = False ) ]
                        else:
                            lines1 += [ TableLine( columns = ['\multirow{%d}{*}{'%(len(tuning_names))+etabins_str[etaBinIdx]+'}',tuning_names[idx], 'Reference'] + ref_values   , 
                                                _contextManaged = False ) ]
                            lines1 += [ TableLine( columns = ['', tuning_names[idx+1], 'Cross Validation'] + cv_values    , _contextManaged = False ) ]
      
                    lines1 += [ HLine(_contextManaged = False) ]
                lines1 += [ HLine(_contextManaged = False) ]
      

                ### Calculate the final efficiencies
                lines2 = []
                lines2 += [ HLine(_contextManaged = False) ]
                lines2 += [ HLine(_contextManaged = False) ]
                lines2 += [ TableLine( columns = ['',colorPD+r'$P_{D}[\%]$',colorPF+r'$F_{a}[\%]$'], _contextManaged = False ) ]
                lines2 += [ HLine(_contextManaged = False) ]
                for idx, tag in enumerate( train_tags ):
                    itable = self.integrate( best_inits, tag )
                    pd = itable[operation_point+'_pd_op'].values[0]*100
                    pd_std = itable[operation_point+'_pd_op'].values[1]*100
                    fa = itable[operation_point+'_fa_op'].values[0]*100
                    fa_std = itable[operation_point+'_fa_op'].values[1]*100
                    pdref = itable[operation_point+'_pd_ref'].values[0]*100
                    faref = itable[operation_point+'_fa_ref'].values[0]*100
                    if idx > 0:
                        lines2 += [ TableLine( columns = [tag ,colorPD+('%1.2f$\pm$%1.2f')%(pd,pd_std),colorPF+('%1.2f$\pm$%1.2f')%(fa,fa_std) ], _contextManaged = False ) ]
                    else:
                      lines2 += [ TableLine( columns = ['Ref.' ,colorPD+('%1.2f')%(pdref),colorPF+('%1.2f')%(faref) ], _contextManaged = False ) ]
                      lines2 += [ TableLine( columns = [tag ,colorPD+('%1.2f$\pm$%1.2f')%(pd,pd_std),colorPF+('%1.2f$\pm$%1.2f')%(fa,fa_std) ], _contextManaged = False ) ]


                # Create all tables into the PDF Latex 
                #with BeamerSection( name = operation_point.replace('_','\_') ):
                with BeamerSlide( title = "The Cross Validation Efficiency Values For All Tunings"  ):          
                    with Table( caption = 'The $P_{d}$, $F_{a}$ and $SP $ values for each phase space for each method.') as table:
                        with ResizeBox( size = 1. ) as rb:
                            with Tabular( columns = '|lcc|' + 'ccc|' * len(etbins_str) ) as tabular:
                                tabular = tabular
                                for line in lines1:
                                    if isinstance(line, TableLine):
                                        tabular += line
                                    else:
                                        TableLine(line, rounding = None)

                with BeamerSlide( title = "The General Efficiency"  ):          
                    with Table( caption = 'The general efficiency for the cross validation method for each method.') as table:
                        with ResizeBox( size = 0.7 ) as rb:
                            with Tabular( columns = 'lc' + 'c' * 2 ) as tabular:
                                tabular = tabular
                                for line in lines2:
                                    if isinstance(line, TableLine):
                                        tabular += line
                                    else:
                                        TableLine(line, rounding = None)




