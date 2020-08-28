#!/usr/local/bin/python3
# New RFI excisor class
# Saves as .ascii zap file

#
import sys
import numpy as np
from pypulse.archive import Archive
import matplotlib.pyplot as plt
import matplotlib.colors as clr
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
        self.subs = self.ar.getNsubint()
        self.tsc = 32
        self.ar.tscrunch(nsubint = self.tsc)
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
            self.omit = self.nn_get_omission( show = True )
            np.set_printoptions( threshold = sys.maxsize )
            print(self.omit)
        else:
            sys.exit()

    def nn_get_omission( self, show = False ):
        pred = np.around( np.squeeze( self.nn.pred_data( self.x ) ), decimals = 0 ).astype(np.int)
        pred = np.array([x==1 for x in pred])
        pred = np.reshape( pred, ( self.ar.getNsubint(), self.ar.getNchan() ) )
        pred = ~pred

        if show:
            fig = plt.figure( figsize = (7, 7) )
            ax = fig.add_subplot(111)
            ax.imshow( pred, cmap = plt.cm.Blues, aspect = 'auto' )
            fig.colorbar( plt.cm.ScalarMappable( norm = clr.Normalize( vmin = 0, vmax = 1 ), cmap = plt.cm.Blues ), ax = ax )
            plt.show()

        return pred


    def get_omission_matrix( self, **kwargs ):

        rms, lin_rms, mu, sigma = u.rms_arr_properties( self.ar.getData(), self.opw, 1.0 ) # Needs to input 2D array

        # Creates the histogram
        plot.histogram_and_curves( lin_rms, mean = mu, std_dev = sigma, bins = (self.ar.getNchan() * self.ar.getNsubint()) // 4, x_axis = 'Root Mean Squared', y_axis = 'Frequency Density', title = r'$M={},\ \sigma={}$'.format( mu, sigma ), **kwargs )

        if self.method == 'chauvenet':
            rej_arr = physics.chauvenet( rms, median = mu, std_dev = sigma, threshold = 2.5 )
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
                    if rej == True:
                        #f.write( str(i) + " " + str(self.ar.freq[i][j]) + "\n" )
                        for k in range( 0 + i*(self.subs//self.tsc), (i+1)*(self.subs//self.tsc) ):
                            f.write( f'{k} {self.ar.freq[k][j]}\n' )
        return outfile

    def save_training( self, outroot = "gaussian_training" ):
        tr_f, val_f = outroot + ".training", outroot + ".validation"
        with open( tr_f, 'w' ) as t:
            t.write( f'# Training set for {self.ar.getName()} taken on {int(self.ar.getMJD())} at {self.ar.getFrontend()}\n' )
        with open( val_f, 'w' ) as t:
            t.write( f'# Validation set for {self.ar.getName()} taken on {int(self.ar.getMJD())} at {self.ar.getFrontend()}\n' )

        k = 0
        for i, t in enumerate( self.ar.pscrunch().getData() ):
            for j, prof in enumerate( t ):
                if self.omit[i][j] == 1:
                    p = np.append( prof, 0 )
                    #inv_p = np.append( -1*prof, 0 )
                elif self.omit[i][j] == 0:
                    p = np.append( prof, 1 )
                    #inv_p = np.append( -1*prof, 1 )
                else:
                    print( f"Rejection array returned nothing for {i},{j}. Skipping." )
                    continue

                if (k+1) % 6 == 0:
                    with open( val_f, 'a' ) as t:
                        np.savetxt( t, p, fmt = '%1.5f ', newline = '' )
                        t.write( "\n" )
                        #np.savetxt( t, inv_p, fmt = '%1.5f ', newline = '' )
                        #t.write( "\n" )
                else:
                    with open( tr_f, 'a' ) as t:
                        np.savetxt( t, p, fmt = '%1.5f ', newline = '' )
                        t.write( "\n" )
                        #np.savetxt( t, inv_p, fmt = '%1.5f ', newline = '' )
                        #t.write( "\n" )
                k += 1




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

    #z = Zap( file, temp, method, nn_params = "J1829+2456", verbose = True, show = True, curve_list = [norm.pdf], x_lims = [0, 20] )
    z = Zap( file, temp, method, nn_params = "J1829+2456", verbose = False, show = True, curve_list = [norm.pdf], x_lims = [0, 50] )
    #z.save( outroot = f"../Zap/{file[6:20]}_lbw_{file[-9:-5]}_2048", ext = '.zap' )
    #z.save_training( outroot = f'{z.ar.getName()}_{int(z.ar.getMJD())}_{z.ar.getFrontend()}_{z.ar.getNbin()}' )
