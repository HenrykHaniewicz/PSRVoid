#!/usr/local/bin/python3
# New RFI excisor class
# Saves as .ascii zap file

#
import sys
import numpy as np
from pypulse.archive import Archive
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# Local imports
import u
import plot
import physics
from nn import NeuralNet

class Zap():

    """
    Master class for zapping data.
    Requires:

    file        -       .FITS (must be PSRFITS v5+ format)

    Optional:

    template    -       ASCII format:       BIN#    Flux           (Required if not doing NN exicison)
    method      -       Either 'chauvenet', 'DMAD' or 'NN'
    verbose     -       Prints more information to the console
    **kwargs    -       Get parsed to plot.histogram_and_curves() or
    """

    def __init__( self, file, template, method = 'chauvenet', nn_params = None, verbose = False, **kwargs ):
        self.file = file
        self.method = method
        self.verbose = verbose
        self.ar = Archive( file, verbose = False )
        self.ar.tscrunch()
        if method != 'NN':
            _, self.template = u.get_data_from_asc( template )
            self.opw = u.get_1D_OPW_mask( self.template, windowsize = 128 )
            self.omit, self.rms_mu, self.rms_sigma = self.get_omission_matrix( **kwargs )
        elif nn_params != None:
            df = pd.DataFrame( np.reshape( self.ar.getData(), (self.ar.getNsubint() * self.ar.getNchan(), self.ar.getNbin()) ) )
            scaler = MinMaxScaler()
            scaled_df = scaler.fit_transform( df.iloc[:, :] )
            scaled_df = pd.DataFrame( scaled_df )
            self.x = scaled_df.iloc[:, :].values.transpose()
            self.nn = NeuralNet( self.x, np.array([[0], [0]]) )
            self.nn.dims = [ self.ar.getNbin(), 20, 1 ]
            self.nn.load_params( root = nn_params )
            self.omit = self.nn_get_omission()
            np.set_printoptions( threshold = sys.maxsize )
            print(self.omit)
        else:
            sys.exit()

    def nn_get_omission( self ):
        pred = np.around( np.squeeze( self.nn.pred_data( self.x, False ) ), decimals = 0 ).astype(np.int)
        pred = np.reshape( pred, ( self.ar.getNsubint(), self.ar.getNchan() ) )

        return pred


    def get_omission_matrix( self, **kwargs ):

        rms, lin_rms, mu, sigma = u.rms_arr_properties( self.ar.getData(), self.opw, 1.0 ) # Needs to input 2D array

        # Creates the histogram
        plot.histogram_and_curves( lin_rms, mean = mu, std_dev = sigma, bins = (self.ar.getNchan() * self.ar.getNsubint()) // 4, x_axis = 'Root Mean Squared', y_axis = 'Frequency Density', title = r'$M={},\ \sigma={}$'.format( mu, sigma ), **kwargs )

        if self.method == 'chauvenet':
            rej_arr = physics.chauvenet( rms, median = mu, std_dev = sigma, threshold = 3 )
        elif self.method == 'DMAD':
            rej_arr = physics.DMAD( lin_rms, threshold = 3.5 )
            rej_arr = np.reshape( rej_arr, ( self.ar.getNsubint(), self.ar.getNchan() ) )

        if self.verbose:
            print( "Rejection criterion created." )

        return rej_arr, mu, sigma

    # Save as ASCII text file
    def save( self, outroot = "zap_out", ext = '.ascii' ):
        outfile = outroot + ext
        with open( outfile, 'w+' ) as f:
            for i, t in enumerate( self.omit ):
                for j, rej in enumerate( t ):
                    if rej == False:
                        f.write( str(i) + " " + str(self.ar.freq[i][j]) + "\n" )
        return outfile



def check_args():
    args = sys.argv
    if len( args ) < 3:
        raise Exception( "Must provide a PSRFITS file and a template" )
    file, temp = args[1:3]
    if len( args ) == 4:
        method = args[3]
    else:
        method = 'chauvenet'

    return file, temp, method

if __name__ == "__main__":

    file, temp, method = check_args()

    z = Zap( file, temp, method, nn_params = "J1829+2456", verbose = False, show = True, curve_list = [norm.pdf], x_lims = [0, 200] )
    #z.save( outroot = f"Zap/zap_{file}", ext = '.nn.ascii' )
