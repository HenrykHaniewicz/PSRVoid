#!/usr/local/bin/python3
# New RFI excisor class
# Saves as .ascii zap file

#
import sys
import numpy as np
from pypulse.archive import Archive
from sklearn.model_selection import train_test_split
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
        if "cal" in self.file:
            raise ValueError( f"File {self.file} is not in PSR format." )
        elif "59071" in self.file:
            raise ValueError( f"Not doing 59071..." )
        self.method = method
        self.verbose = verbose
        self.ar = Archive( file, verbose = False )
        if method != 'NN':
            _, self.template = u.get_data_from_asc( template )
            self.opw = u.get_1D_OPW_mask( self.template, windowsize = 128 )
            self.omit, self.rms_mu, self.rms_sigma = self.get_omission_matrix( **kwargs )
            unique, counts = np.unique( self.omit, return_counts = True )
            print( f"Good channels: {100*(counts[0]/sum(counts)):.3f}%")
            print( f"Bad channels: {100*(counts[1]/sum(counts)):.3f}%")
        elif nn_params != None:
            df = pd.DataFrame( np.reshape( self.ar.getData(), (self.ar.getNsubint() * self.ar.getNchan(), self.ar.getNbin()) ) )
            scaler = MinMaxScaler()
            scaled_df = scaler.fit_transform( df.iloc[:, :] )
            scaled_df = pd.DataFrame( scaled_df )
            self.x = scaled_df.iloc[:, :].values.transpose()
            self.nn = NeuralNet( self.x, np.array([[0], [0]]) )
            self.nn.dims = [ self.ar.getNbin(), 512, 10, 13, 8, 6, 6, 4, 4, 1 ]
            self.nn.threshold = 0.5
            self.nn.load_params( root = nn_params )
            self.omit = self.nn_get_omission()
            np.set_printoptions( threshold = sys.maxsize )
            unique, counts = np.unique( self.omit, return_counts = True )
            print( f"Good channels: {100*(counts[0]/sum(counts)):.3f}%")
            print( f"Bad channels: {100*(counts[1]/sum(counts)):.3f}%")
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
            rej_arr = physics.chauvenet( rms, median = mu, std_dev = sigma, threshold = 2.0 )
        elif self.method == 'DMAD':
            rej_arr = physics.DMAD( lin_rms, threshold = 3.5 )
            rej_arr = np.reshape( rej_arr, ( self.ar.getNsubint(), self.ar.getNchan() ) )

        if self.verbose:
            print( "Rejection criterion created." )

        return rej_arr, mu, sigma

    def plot_mask( self, **kwargs ):

        fig = plt.figure( figsize = (7, 7) )
        ax = fig.add_subplot(111)
        ax.imshow( self.omit.T, cmap = plt.cm.gray, interpolation = 'nearest', aspect = 'auto' )
        plt.show()

    def save_training_set( self, val_size = 0.2 ):
        # From Chauvenet or DMAD. 1 is bad channel

        with open( f'{self.ar.getName()}_{int(self.ar.getMJD())}_{self.ar.getFrontend()}_{self.ar.getNbin()}.training', 'w' ) as t:
            t.write( f'# Training set for {self.ar.getName()} taken on {int(self.ar.getMJD())} at {self.ar.getFrontend()}\n' )
        with open( f'{self.ar.getName()}_{int(self.ar.getMJD())}_{self.ar.getFrontend()}_{self.ar.getNbin()}.validation', 'w' ) as t:
            t.write( f'# Validation set for {self.ar.getName()} taken on {int(self.ar.getMJD())} at {self.ar.getFrontend()}\n' )

        ps_0 = np.zeros(2049)[np.newaxis, :]
        ps_1 = np.zeros(2049)[np.newaxis, :]

        d = self.ar.getData().reshape( ( self.ar.getNchan() * self.ar.getNsubint(), self.ar.getNbin() ) )
        omission = self.omit.reshape( ( self.ar.getNchan() * self.ar.getNsubint() ) )

        i = 1
        for omit, profile in zip( omission, d ):
            try:
                choice = int(omit)
                if choice == 1:
                    choice = 0
                elif choice == 0:
                    choice = 1
            except ValueError:
                choice = -1

            print( i, end = '\r' )

            if choice != -1:
                # Creates the profile / choice pairs and doubles up with the reciprocal profiles.
                p = np.append( profile, choice )
                #inv_p = np.append( -1*profile, choice )
                if choice == 0:
                    ps_0 = np.append( ps_0, p[np.newaxis, :], axis = 0 )
                else:
                    ps_1 = np.append( ps_1, p[np.newaxis, :], axis = 0 )

            i += 1

        ps_0, ps_1 = np.delete( ps_0, 0, 0 ), np.delete( ps_1, 0, 0 )

        # Sort into training / validation sets
        train, validation = train_test_split( ps_0, test_size = val_size )
        ones_t, ones_v = train_test_split( ps_1, test_size = val_size )
        train, validation = np.append( train, ones_t, axis = 0 ), np.append( validation, ones_v, axis = 0 )

        np.random.shuffle( train ), np.random.shuffle( validation )

        for k in train:
            with open( f'{self.ar.getName()}_{int(self.ar.getMJD())}_{self.ar.getFrontend()}_{self.ar.getNbin()}.training', 'a' ) as t:
                np.savetxt( t, k, fmt = '%1.5f ', newline = '' )
                t.write( "\n" )
                #np.savetxt( t, inv_p, fmt = '%1.5f ', newline = '' )
                #t.write( "\n" )

        for k in validation:
            with open( f'{self.ar.getName()}_{int(self.ar.getMJD())}_{self.ar.getFrontend()}_{self.ar.getNbin()}.validation', 'a' ) as t:
                np.savetxt( t, k, fmt = '%1.5f ', newline = '' )
                t.write( "\n" )



    # Save as ASCII text file
    def save( self, outroot = "zap_out", ext = '.ascii' ):
        outfile = outroot + ext
        with open( outfile, 'w+' ) as f:
            for i, t in enumerate( self.omit ):
                for j, rej in enumerate( t ):
                    if rej == True:
                        f.write( str(i) + " " + str(self.ar.freq[i][j]) + "\n" )
                        #f.write( f'{k} {self.ar.freq[k][i]}\n' )
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

    z = Zap( file, temp, method, nn_params = "J1829+2456", verbose = True, show = True, curve_list = [norm.pdf], x_lims = [0, 500] )
    z.plot_mask()
    #z.save_training_set()
    z.save( outroot = f"../zap/{file.split('.')[0]}", ext = '.zap' )
