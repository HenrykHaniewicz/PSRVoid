#!/usr/local/bin/python3
# Neural Net class and methods for use in RFI exicion
# Henryk T. Haniewicz, 2019

# External imports
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

# Local imports
from plot import plot_cf
from physics import sigmoid, dsigmoid, dtanh, relu, drelu, lrelu, dlrelu, softmax, dsoftmax, Swish, dSwish

'''
Can also use this file with the syntax ./nn.py training_file validation_file to train the NN
'''

plt.rc( 'text', usetex = True )
plt.rc( 'font', family = 'serif' )

class NeuralNet:
    def __init__( self, x, y, lr = 0.003, verbose = True ):
        self.verbose = verbose
        self.X = x
        self.Y = y
        self.Yh = np.zeros( (1, self.Y.shape[1]) )
        self.dims = [len(x), len(x)//4, 1]
        self.param = {}
        self.ch = {}
        self.grad = {}
        self.loss = []
        self.lr = lr
        self.lr_str = str(lr).replace( '.', '_' )
        self.sam = self.Y.shape[1]
        self.threshold = 0.5

    def seed_init( self ):
        np.random.seed(1)
        for i in range( len( self.dims ) - 1 ):
            self.param[f'W{i+1}'] = np.random.randn(self.dims[i+1], self.dims[i]) / np.sqrt(self.dims[i])
            self.param[f'b{i+1}'] = np.zeros((self.dims[i+1], 1))
        return

    def feed_forward( self ):
        Z1 = self.param['W1'].dot(self.X) + self.param['b1']
        A1 = relu(Z1) # Subject to change
        #A1 = Swish(Z1)
        self.ch['Z1'], self.ch['A1'] = Z1, A1

        if len(self.dims) != 3:
            for i in range( 2, len( self.dims ) - 1 ):
                self.ch[ f'Z{i}' ] = self.param[ f'W{i}' ].dot(self.ch[ f'A{i-1}' ]) + self.param[ f'b{i}' ]
                self.ch[ f'A{i}' ] = relu( self.ch[ f'Z{i}' ] )

        self.ch[ f'Z{len( self.dims ) - 1}' ] = self.param[ f'W{len( self.dims ) - 1}' ].dot(self.ch[ f'A{len( self.dims ) - 2}' ]) + self.param[ f'b{len( self.dims ) - 1}' ]
        self.ch[ f'A{len( self.dims ) - 1}' ] = sigmoid( self.ch[ f'Z{len( self.dims ) - 1}' ] )
        #self.ch[ f'A{len( self.dims ) - 1}' ] = softmax( self.ch[ f'Z{len( self.dims ) - 1}' ] )
        #self.ch[ f'A{len( self.dims ) - 1}' ] = 0.5*(np.tanh( self.ch[ f'Z{len( self.dims ) - 1}' ] ) + 1)

        self.Yh = self.ch[ f'A{len( self.dims ) - 1}' ]
        try:
            loss = self.nloss( self.Yh )
        except ValueError:
            loss = None
        return self.Yh, loss

    def nloss( self, Yh ):
        loss = ( 1./self.sam ) * ( -np.dot( self.Y, np.log(Yh).T ) - np.dot( 1 - self.Y, np.log( 1 - Yh ).T ) )
        return loss

    def back_prop( self ):
        dLoss_Z, dLoss_A, dLoss_W, dLoss_b = np.zeros( len( self.dims ) - 1, dtype = object ), np.zeros( len( self.dims ) - 1, dtype = object ), np.zeros( len( self.dims ) - 1, dtype = object ), np.zeros( len( self.dims ) - 1, dtype = object )
        dLoss_A[-1] = -( np.divide( self.Y, self.Yh ) - np.divide( 1 - self.Y, 1 - self.Yh ) )

        for i in np.arange( len( self.dims ) - 2, 0, -1 ):
            if (i == len( self.dims ) - 2):
                dLoss_Z[i] = dLoss_A[i] * dsigmoid( self.ch[ f'Z{i+1}' ] )
                #dLoss_Z[i] = dLoss_A[i] * np.diag(dsoftmax( self.ch[ f'Z{i+1}' ] ))
                #dLoss_Z[i] = dLoss_A[i] * 0.5 * dtanh( self.ch[ f'Z{i+1}' ] )
                dLoss_A[i-1] = np.dot( self.param[ f'W{i+1}' ].T, dLoss_Z[i] )
            else:
                dLoss_Z[i] = dLoss_A[i] * drelu( self.ch[ f'Z{i+1}' ] )
                dLoss_A[i-1] = np.dot( self.param[ f'W{i+1}' ].T, dLoss_Z[i] )
            dLoss_W[i] = 1./self.ch[ f'A{i}' ].shape[1] * np.dot( dLoss_Z[i], self.ch[ f'A{i}' ].T )
            dLoss_b[i] = 1./self.ch[ f'A{i}' ].shape[1] * np.dot( dLoss_Z[i], np.ones( [dLoss_Z[i].shape[1], 1] ) )

        dLoss_Z1 = dLoss_A[0] * drelu( self.ch[ f'Z1' ] ) # Subject to change
        #dLoss_Z1 = dLoss_A[0] * dSwish( self.ch[ f'Z1' ] )
        dLoss_A0 = np.dot( self.param[ f'W1' ].T, dLoss_Z1 )
        dLoss_W1 = 1./self.X.shape[1] * np.dot( dLoss_Z1, self.X.T )
        dLoss_b1 = 1./self.X.shape[1] * np.dot( dLoss_Z1, np.ones( [dLoss_Z1.shape[1],1] ) )

        self.param[ 'W1' ] = self.param[ 'W1' ] - self.lr * dLoss_W1
        self.param[ 'b1' ] = self.param[ 'b1' ] - self.lr * dLoss_b1
        for i in np.arange( 1, len( self.dims ) - 1 ):
            self.param[ f'W{i+1}' ] = self.param[ f'W{i+1}' ] - self.lr * dLoss_W[i]
            self.param[ f'b{i+1}' ] = self.param[ f'b{i+1}' ] - self.lr * dLoss_b[i]

        return

    def pred_train( self, x, y ):
        self.X = x
        self.Y = y
        comp = np.zeros( ( 1, x.shape[1] ) )
        pred, loss = self.feed_forward()

        for i in range( 0, pred.shape[1] ):
            if pred[ 0, i ] > self.threshold:
                comp[ 0, i ] = 1
            else:
                comp[ 0, i ] = 0

        if self.verbose:
            print( f"Acc: {np.sum( (comp == y)/x.shape[1] )}" )
            '''c11 = np.logical_and( comp, y )
            c00 = np.logical_not( np.logical_or( comp, y ) )
            print( f"Sens: {c11 / np.sum((y == 1))}" )
            print( f"Spec: {c00 / np.sum((y == 0))}" )
            print( f"Prec: {c11 / np.sum((comp == 1))}" )'''


        return comp

    def pred_data( self, x ):
        self.X = x
        comp = np.zeros( ( 1, x.shape[1] ) )
        pred, loss = self.feed_forward()

        for i in range( 0, pred.shape[1] ):
            if pred[ 0, i ] > self.threshold:
                comp[ 0, i ] = 1
            else:
                comp[ 0, i ] = 0

        if self.verbose:
            print( comp )

        return comp

    def gd( self, iter = 3000 ):
        if self.verbose:
            print( self.dims )
            print( f'Lr = {self.lr}' )

        np.random.seed(1)

        self.seed_init()

        for i in range( 0, iter ):
            Yh, loss = self.feed_forward()
            self.back_prop()

            if i % 50 == 0:
                print( f"Cost after iteration {i}: {loss}" )
                self.loss.append( loss )

        plt.plot( np.squeeze( self.loss ), color = 'k' )
        plt.ylabel( 'Loss' )
        plt.xlabel( r'Iteration ($\times$50)' )
        plt.title( f"Learning rate = {str( self.lr )}" )
        plt.savefig( f"/Volumes/Backup_Plus/DATA/J1829+2456/1829+2456_2019/RFI_test/nn_{self.lr_str}_relu_tanh.pdf", format = 'pdf' )
        plt.show()

        return

    def save_params( self, root = 'nn_params' ):
        for i in range( len( self.dims ) - 1 ):
            np.save( f'{root}.w{i+1}.nn.params', self.param[f'W{i+1}'] )
            np.save( f'{root}.b{i+1}.nn.params', self.param[f'b{i+1}'] )

    def load_params( self, root = 'nn_params' ):
        for i in range( len( self.dims ) - 1 ):
            self.param[f'W{i+1}'] = np.load( f'../1829+2456_2018_0002_densecampaign/NN/{root}.w{i+1}.nn.params.npy' )
            self.param[f'b{i+1}'] = np.load( f'../1829+2456_2018_0002_densecampaign/NN/{root}.b{i+1}.nn.params.npy' )

def read_data( training, validation ):

    df_t = pd.read_table( training, header = None, sep = '\s+', skiprows = 1 )
    df_t = df_t.astype( float )
    df_v = pd.read_table( validation, header = None, sep = '\s+', skiprows = 1 )
    df_v = df_v.astype( float )

    # Training data setup
    scaled_df_t = df_t
    names = df_t.columns[0:-1]
    scaler = MinMaxScaler()
    scaled_df_t = scaler.fit_transform( df_t.iloc[:, 0:-1] )
    scaled_df_t = pd.DataFrame( scaled_df_t, columns = names )
    x = scaled_df_t.iloc[:, :].values.transpose()
    y = df_t.iloc[:, -1:].values.transpose()

    # Validation data setup
    scaled_df_v = df_v
    names = df_v.columns[0:-1]
    scaler = MinMaxScaler()
    scaled_df_v = scaler.fit_transform( df_v.iloc[:, 0:-1] )
    scaled_df_v = pd.DataFrame( scaled_df_v, columns = names )
    xv = scaled_df_v.iloc[:, :].values.transpose()
    yv = df_v.iloc[:, -1:].values.transpose()

    return x, y, xv, yv

if __name__ == "__main__":

    SAVE = True

    if len(sys.argv) == 1:
        print('''Trains the neural network

    Use: nn.py [training file] [validation file]
        ''')
        exit(1)

    training, validation = sys.argv[1], sys.argv[2]
    x, y, xv, yv = read_data( training, validation )

    # Start neural network
    nn = NeuralNet( x, y, 0.0005 )
    nn.dims = [ len(x), 16, 16, 16, 16, 1 ]
    nn.threshold = 0.5
    nn.gd( iter = 100000 )

    pred_train = nn.pred_train( x, y )
    print("\n")
    pred_valid = nn.pred_train( xv, yv )

    nn.X, nn.Y = x, y
    target = np.around( np.squeeze( y ), decimals = 0 ).astype(np.int)
    predicted = np.around( np.squeeze( pred_train ), decimals = 0 ).astype(np.int)
    plot_cf( target, predicted, f'Accuracy: {100*np.sum( (pred_train == y)/x.shape[1] ):.2f}$\%$', save = f"/Volumes/Backup_Plus/DATA/J1829+2456/1829+2456_2019/RFI_test/nn_{nn.lr_str}_relu_tanh_cf_train.pdf" )

    nn.X, nn.Y = xv, yv
    target = np.around( np.squeeze( yv ), decimals = 0 ).astype(np.int)
    predicted = np.around( np.squeeze( pred_valid ), decimals = 0 ).astype(np.int)
    plot_cf( target, predicted, 'Validation Set', save = f"/Volumes/Backup_Plus/DATA/J1829+2456/1829+2456_2019/RFI_test/nn_{nn.lr_str}_relu_tanh_cf_val.pdf" )

    if SAVE:
        nn.save_params( root = 'J1829+2456' )
