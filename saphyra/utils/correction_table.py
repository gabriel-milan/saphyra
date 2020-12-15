
from Gaugi.monet.AtlasStyle import *
from Gaugi.monet.PlotFunctions import *
from Gaugi.monet.TAxisFunctions import *

from ROOT import kBlack,kBlue,kRed,kAzure,kGreen,kMagenta,kCyan,kOrange,kGray,kYellow,kWhite
from Gaugi.monet import *
from array import array
from copy import deepcopy
import time,os,math,sys,pprint,glob,warnings
import numpy as np
import ROOT, math
import ctypes


class correction_table(Logger):

    def __init__(self, generator, etbins, etabins, x_bin_size, y_bin_size, ymin, ymax, ):

        self.__etbins = etbins
        self.__etabins = etabins
        self.__ymin = ymin
        self.__ymax = ymax
        self.__x_bin_size = x_bin_size
        self.__y_bin_size = y_bin_size




    def fill( self, generator, paths, models, references ):


        # make template dataframe
        dataframe = collections.OrderedDict({
                      'name':[],
                      'reference_passed':[],
                      'reference_total':[],
                      'reference_eff':[],
                      'et_bin':[],
                      'eta_bin':[],
                      'signal_passed':[],
                      'signal_total':[],
                      'signal_eff':[],
                      'background_passed':[],
                      'backgrond_total':[],
                      'backgrond_eff':[],
                      'signal_corrected_passed':[],
                      'signal_corrected_total':[],
                      'signal_corrected_eff':[],
                      'background_corrected_passed':[],
                      'background_corrected_total':[],
                      'background_corrected_eff':[],
                      'th2_signal':[],
                      'th2_background':[]
                     })

        # reduce verbose
        def add(key,value):
          dataframe[key].append(value)


        # Loop over all et/eta bins
        for et_bin in range(paths):
            for eta_bin in range(paths):

                path = paths[et_bin][eta_bin]
                data, target, avgmu, references = generator(path)
                model = models[et_bin][eta_bin]
                model['thresholds'] = {}

                # Get output tensor and convert to numpy
                outputs = model['model'](data).numpy()

                # Get all limits using the output
                xmin = int(np.percentile(outputs , 0.2))
                xmax = int(np.percentile(outputs, 0.98))
                xbins = (xmax-xmin)/self.__x_bin_size
                ybins = (self.__ymax-self.__ymin)/self.__y_bin_size


                from ROOT import TH2F
                th2_signal = TH2F( 'th2_signal', xbins, xmin, xmax, ybins, self.__ymin, self.__ymax )
                th2_signal.FillN( outputs[target==1], avgmu[target==1], 1)
                th2_background = TH2F( 'th2_background', xbins, xmin, xmax, ybins, self.__ymin, self.__ymax )
                th2_background.FillN( outputs[target==0], avgmu[target==0], 1)

                for name, ref in references.items():

                    reference_num = ref['passed']
                    reference_den = ref['total']
                    target = reference_num/reference_den


                    false_alarm = 1.0
                    while false_alarm > false_alarm_limit:

                        threshold = self.find_threshold( th2_signal.ProjectionX(), target )
                        # Get the efficiency without linear adjustment
                        eff, num, den = self.calculate_num_and_den(th2_signal, reference_eff, 0.0, target)

                        # Apply the linear adjustment and fix it in case of positive slope
                        slope, offset = self.fit( th2_signal, value )
                        slope = 0 if slope>0 else slope
                        offset = threshold if slope>0 else offset
                        if slope>0:
                          MSG_WARNING(self, "Retrieved positive angular factor of the linear correction, setting to 0!")

                        # Get the efficiency with linear adjustment
                        signal_eff, signal_num, signal_den = self.calculate_num_and_den(th2_signal, target, slope, offset)
                        background_eff, background_num, background_den = self.calculate_num_and_den(th2_background, target, slope, offset)

                        false_alarm = background_num/background_den # get the passed/total

                        if false_alarm > false_alarm_limit:
                            # Reduce the reference value by hand
                            value-=0.0025

                    MSG_INFO( self, 'Signal with correction is: %1.2f%%', signal_num/signal_den * 100 )
                    MSG_INFO( self, 'Background with correction is: %1.2f%%', background_num/background_den * 100 )


                    model['thresholds'][name] = {'offset':offset, 'slope':slope, 'offset_noadjust' : threshold}

                    # Save some values into the main table
                    add( 'name'                        , name )
                    add( 'et_bin'                      , et_bin  )
                    add( 'eta_bin'                     , eta_bin )
                    add( 'reference_passed'            , reference_num )
                    add( 'reference_total'             , reference_den )
                    add( 'reference_eff'               , reference_num/reference_den )
                    add( 'signal_passed'               , signal_num )
                    add( 'signal_total'                , signal_den )
                    add( 'signal_eff'                  , signal_num/signal_den )
                    add( 'background_passed'           , background_num )
                    add( 'background_total'            , background_den )
                    add( 'background_eff'              , background_num/background_den )
                    add( 'signal_corrected_passed'     , signal_corrected_num )
                    add( 'signal_corrected_total'      , signal_corrected_den )
                    add( 'signal_corrected_eff'        , signal_corrected_num/signal_corrected_den )
                    add( 'background_corrected_passed' , background_corrected_num )
                    add( 'background_corrected_total'  , background_corrected_den )
                    add( 'background_corrected_eff'    , background_corrected_num/background_corrected_den )
                    add( 'th2_signal'                  , th2_signal )
                    add( 'th2_background'              , th2_background )



    # Export all models ringer
    def export( self, models, model_output_format , conf_output, reference_name, to_onnx=False):


        from ROOT import TEnv

        model_etmin_vec = []
        model_etmax_vec = []
        model_etamin_vec = []
        model_etamax_vec = []
        model_paths = []

        slopes = []
        offsets = []

        # serialize all models
        for model in models:

            model_etmin_vec.append( model['etBin'][0] )
            model_etmax_vec.append( model['etBin'][1] )
            model_etamin_vec.append( model['etaBin'][0] )
            model_etamax_vec.append( model['etaBin'][1] )

            etBinIdx = model['etBinIdx']
            etaBinIdx = model['etaBinIdx']

            model_name = model_output_format%( etBinIdx, etaBinIdx )
            model_paths.append( model_name )

            # Save onnx mode!
            if to_onnx:
                import onnx, keras2onnx
                onnx_model = keras2onnx.convert_keras(model['model'], model['model'].name)
                onnx.save_model(onnx_model, model_name+'.onnx')

            model_json = model['model'].to_json()
            with open(model_name+".json", "w") as json_file:
                json_file.write(model_json)
                # saving the model weight separately
                model['model'].save_weights(model_name+".h5")

            slopes.append( model['thresholds'][reference_name]['slope'] )
            offsets.append( model['thresholds'][reference_name]['offsets'] )


        def list_to_str( l ):
            s = str()
            for ll in l:
              s+=str(ll)+'; '
            return s[:-2]

        # Write the config file
        file = TEnv( 'ringer' )
        file.SetValue( "__name__", 'should_be_filled' )
        file.SetValue( "__version__", 'should_be_filled' )
        file.SetValue( "__operation__", reference_name )
        file.SetValue( "__signature__", 'should_be_filled' )
        file.SetValue( "Model__size"  , str(len(models)) )
        file.SetValue( "Model__etmin" , list_to_str(model_etmin_vec) )
        file.SetValue( "Model__etmax" , list_to_str(model_etmax_vec) )
        file.SetValue( "Model__etamin", list_to_str(model_etamin_vec) )
        file.SetValue( "Model__etamax", list_to_str(model_etamax_vec) )
        file.SetValue( "Model__path"  , list_to_str( model_paths ) )
        file.SetValue( "Threshold__size"  , str(len(models)) )
        file.SetValue( "Threshold__etmin" , list_to_str(model_etmin_vec) )
        file.SetValue( "Threshold__etmax" , list_to_str(model_etmax_vec) )
        file.SetValue( "Threshold__etamin", list_to_str(model_etamin_vec) )
        file.SetValue( "Threshold__etamax", list_to_str(model_etamax_vec) )
        file.SetValue( "Threshold__slope" , list_to_str(slopes) )
        file.SetValue( "Threshold__offset", list_to_str(offsets) )
        file.SetValue( "Threshold__MaxAverageMu", 100)
        file.WriteFile(conf_output)




    #
    # Find the threshold given a reference value
    #
    def find_threshold(self, th1,effref):
        nbins = th1.GetNbinsX()
        fullArea = th1.Integral(0,nbins+1)
        if fullArea == 0:
            return 0,1
        notDetected = 0.0; i = 0
        while 1. - notDetected > effref:
            cutArea = hist.Integral(0,i)
            i+=1
            prevNotDetected = notDetected
            notDetected = cutArea/fullArea
        eff = 1. - notDetected
        prevEff = 1. -prevNotDetected
        deltaEff = (eff - prevEff)
        threshold = th1.GetBinCenter(i-1)+(effref-prevEff)/deltaEff*(th1.GetBinCenter(i)-th1.GetBinCenter(i-1))
        error = 1./math.sqrt(fullArea)
        return threshold

    #
    # Get all points in the 2D histogram given a reference value
    #
    def get_points( self, th2 , effref):
        nbinsy = th2.GetNbinsY()
        x = list(); y = list(); errors = list()
        for by in range(nbinsy):
            xproj = th2.ProjectionX('xproj'+str(time.time()),by+1,by+1)
            discr, error = self.find_threshold(xproj,effref)
            dbin = xproj.FindBin(discr)
            x.append(discr); y.append(th2.GetYaxis().GetBinCenter(by+1))
            errors.append( error )
        return x,y,errors



    #
    # Calculate the linear fit given a 2D histogram and reference value and return the slope and offset
    #
    def fit(self, th2,effref):
        x_points, y_points, error_points = self.get_points(th2, effref )
        import array
        g = ROOT.TGraphErrors( len(discr_points)
                             , array.array('d',y_points,)
                             , array.array('d',x_points)
                             , array.array('d',[0.]*len(x_points))
                             , array.array('d',error_points) )
        firstBinVal = th2.GetYaxis().GetBinLowEdge(th2.GetYaxis().GetFirst())
        lastBinVal = th2.GetYaxis().GetBinLowEdge(th2.GetYaxis().GetLast()+1)
        f1 = ROOT.TF1('f1','pol1',firstBinVal, lastBinVal)
        g.Fit(f1,"FRq")
        slope = f1.GetParameter(1)
        offset = f1.GetParameter(0)
        return slope, offset


    #
    # Calculate the numerator and denomitator given the 2D histogram and slope/offset parameters
    #
    def calculate_num_and_den(th2, slope, offset) :

      nbinsy = th2.GetNbinsY()
      th1_num = th2.ProjectionY(th2.GetName()+'_proj'+str(time.time()),1,1)
      th1_num.Reset("ICESM")
      numerator=0; denominator=0
      # Calculate how many events passed by the threshold
      for by in range(nbinsy) :
          xproj = th2.ProjectionX('xproj'+str(time.time()),by+1,by+1)
          # Apply the correction using ax+b formula
          threshold = slope*th2.GetYaxis().GetBinCenter(by+1)+ offset
          dbin = xproj.FindBin(threshold)
          num = xproj.Integral(dbin+1,xproj.GetNbinsX()+1)
          th1_num.SetBinContent(by+1,num)
          numerator+=num
          denominator+=xproj.Integral(-1, xproj.GetNbinsX()+1)

      # Calculate the efficiency histogram
      th1_den = th2.ProjectionY(th2.GetName()+'_proj'+str(time.time()),1,1)
      th1_eff = th1_num.Clone()
      th1_eff.Divide(th1_den)
      # Fix the error bar
      for bx in range(th1_eff.GetNbinsX()):
          if th1_den.GetBinContent(bx+1) != 0 :
              eff = th1_eff.GetBinContent(bx+1)
              try:
                  error = math.sqrt(eff*(1-eff)/th1_den.GetBinContent(bx+1))
              except:
                  error=0
              th1_eff.SetBinError(bx+1,eff)
          else:
              th1_eff.SetBinError(bx+1,0)

      return th1_eff, numerator, denominator







#def Plot2DHist( chist, hist2D, a, b, discr_points, nvtx_points, error_points, outname, xlabel='',
#                etBinIdx=None, etaBinIdx=None, etBins=None,etaBins=None):
#
#  from ROOT import TCanvas, gStyle, TLegend, kRed, kBlue, kBlack,TLine,kBird, kOrange,kGray
#  from ROOT import TGraphErrors,TF1,TColor
#  gStyle.SetPalette(kBird)
#  ymax = chist.ymax(); ymin = chist.ymin()
#  xmin = ymin; xmax = ymax
#  drawopt='lpE2'
#  canvas = TCanvas('canvas','canvas',500, 500)
#  canvas.SetRightMargin(0.15)
#  #hist2D.SetTitle('Neural Network output as a function of nvtx, '+partition_name)
#  hist2D.GetXaxis().SetTitle('Neural Network output (Discriminant)')
#  hist2D.GetYaxis().SetTitle(xlabel)
#  hist2D.GetZaxis().SetTitle('Count')
#  #if not removeOutputTansigTF:  hist2D.SetAxisRange(-1,1, 'X' )
#  hist2D.Draw('colz')
#  canvas.SetLogz()
#  import array
#  g1 = TGraphErrors(len(discr_points), array.array('d',discr_points), array.array('d',nvtx_points), array.array('d',error_points)
#                   , array.array('d',[0]*len(discr_points)))
#  g1.SetLineWidth(1)
#  g1.SetLineColor(kBlue)
#  g1.SetMarkerColor(kBlue)
#  g1.Draw("P same")
#  l3 = TLine(b+a*xmin,ymin, a*xmax+b, ymax)
#  l3.SetLineColor(kBlack)
#  l3.SetLineWidth(2)
#  l3.Draw()
#  AddTopLabels2( canvas, etlist=etBins,etalist=etaBins,etidx=etBinIdx,etaidx=etaBinIdx)
#  FormatCanvasAxes(canvas, XLabelSize=16, YLabelSize=16, XTitleOffset=0.87, ZLabelSize=14,ZTitleSize=14, YTitleOffset=0.87, ZTitleOffset=1.1)
#  SetAxisLabels(canvas,'Neural Network output (Discriminant)',xlabel)
#  #AtlasTemplate1(canvas,atlaslabel='Internal')
#  canvas.SaveAs(outname+'.pdf')
#  canvas.SaveAs(outname+'.C')
#  return outname+'.pdf'
#
#
#
#def PlotEff( chist, hist_eff, hist_eff_corr, refvalue, outname, xlabel=None, runLabel=None,
#            etBinIdx=None, etaBinIdx=None, etBins=None,etaBins=None):
#
#  from ROOT import TCanvas, gStyle, TLegend, kRed, kBlue, kBlack,TLine,kBird, kOrange,kGray
#  from ROOT import TGraphErrors,TF1,TColor
#  gStyle.SetPalette(kBird)
#  ymax = chist.ymax(); ymin = chist.ymin()
#  xmin = ymin; xmax = ymax
#  drawopt='lpE2'
#
#
#  canvas = TCanvas('canvas','canvas',500, 500)
#  #hist_eff.SetTitle('Signal Efficiency in: '+partition_name)
#  hist_eff.SetLineColor(kGray+2)
#  hist_eff.SetMarkerColor(kGray+2)
#  hist_eff.SetFillColor(TColor.GetColorTransparent(kGray, .5))
#  hist_eff_corr.SetLineColor(kBlue+1)
#  hist_eff_corr.SetMarkerColor(kBlue+1)
#  hist_eff_corr.SetFillColor(TColor.GetColorTransparent(kBlue+1, .5))
#  AddHistogram(canvas,hist_eff,drawopt)
#  AddHistogram(canvas,hist_eff_corr,drawopt)
#  l0 = TLine(xmin,refvalue,xmax,refvalue)
#  l0.SetLineColor(kBlack)
#  l0.Draw()
#  l1 = TLine(xmin,refvalue,xmax,refvalue)
#  l1.SetLineColor(kGray+2)
#  l1.SetLineStyle(9)
#  l1.Draw()
#  AddTopLabels1( canvas, ['Without correction','With correction'], runLabel=runLabel, legOpt='p',
#                etlist=etBins,
#                etalist=etaBins,
#                etidx=etBinIdx,etaidx=etaBinIdx)
#
#  FormatCanvasAxes(canvas, XLabelSize=18, YLabelSize=18, XTitleOffset=0.87, YTitleOffset=1.5)
#  SetAxisLabels(canvas,xlabel,'#epsilon('+xlabel+')')
#  FixYaxisRanges(canvas, ignoreErrors=False,yminc=-eps)
#  AutoFixAxes(canvas,ignoreErrors=False)
#  AddBinLines(canvas,hist_eff)
#  canvas.SaveAs(outname+'.pdf')
#  canvas.SaveAs(outname+'.C')
#  return outname+'.pdf'




