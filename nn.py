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
from physics import sigmoid, dsigmoid, relu, drelu

'''
Can also use this file with the syntax ./nn.py training_file validation_file to train the NN
'''

class NeuralNet:
    def __init__( self, x, y, lr = 0.003 ):
        self.debug = 0;
        self.X = x
        self.Y = y
        self.Yh = np.zeros( (1, self.Y.shape[1]) )
        self.L = 2
        self.dims = [10, 16, 16, 1]
        self.param = {}
        self.ch = {}
        self.grad = {}
        self.loss = []
        self.lr = lr
        self.sam = self.Y.shape[1]
        self.threshold = 0.5
        self.outfile = "nn_params.nn.param"

    def seed_init( self ):
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0])
        self.param['b1'] = np.zeros((self.dims[1], 1))
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1])
        self.param['b2'] = np.zeros((self.dims[2], 1))
        return

    def feed_forward( self ):
        Z1 = self.param['W1'].dot(self.X) + self.param['b1']
        A1 = relu(Z1)
        self.ch['Z1'], self.ch['A1'] = Z1, A1

        Z2 = self.param['W2'].dot(A1) + self.param['b2']
        A2 = sigmoid(Z2)
        self.ch['Z2'], self.ch['A2'] = Z2, A2

        self.Yh = A2
        try:
            loss = self.nloss(A2)
        except ValueError:
            loss = None
        return self.Yh, loss

    def nloss( self, Yh ):
        loss = ( 1./self.sam ) * ( -np.dot( self.Y, np.log(Yh).T ) - np.dot( 1 - self.Y, np.log( 1 - Yh ).T ) )
        return loss

    def back_prop( self ):
        dLoss_Yh = -( np.divide( self.Y, self.Yh ) - np.divide( 1 - self.Y, 1 - self.Yh ) )

        dLoss_Z2 = dLoss_Yh * dsigmoid( self.ch['Z2'] )
        dLoss_A1 = np.dot( self.param["W2"].T, dLoss_Z2 )
        dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot( dLoss_Z2, self.ch['A1'].T )
        dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot( dLoss_Z2, np.ones( [dLoss_Z2.shape[1],1] ) )

        dLoss_Z1 = dLoss_A1 * drelu( self.ch['Z1'] )
        dLoss_A0 = np.dot( self.param["W1"].T, dLoss_Z1 )
        dLoss_W1 = 1./self.X.shape[1] * np.dot( dLoss_Z1, self.X.T )
        dLoss_b1 = 1./self.X.shape[1] * np.dot( dLoss_Z1, np.ones( [dLoss_Z1.shape[1],1] ) )

        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2

        return

    def pred_train( self, x, y, verbose = False ):
        self.X = x
        self.Y = y
        comp = np.zeros( ( 1, x.shape[1] ) )
        pred, loss = self.feed_forward()

        for i in range( 0, pred.shape[1] ):
            if pred[ 0, i ] > self.threshold:
                comp[ 0, i ] = 1
            else:
                comp[ 0, i ] = 0

        if verbose:
            print( f"Acc: {np.sum( (comp == y)/x.shape[1] )}" )

        return comp

    def pred_data( self, x, verbose = False ):
        self.X = x
        comp = np.zeros( ( 1, x.shape[1] ) )
        pred, loss = self.feed_forward()

        for i in range( 0, pred.shape[1] ):
            if pred[ 0, i ] > self.threshold:
                comp[ 0, i ] = 1
            else:
                comp[ 0, i ] = 0

        if verbose:
            print( comp )

        return comp

    def gd( self, iter = 3000 ):
        np.random.seed(1)

        self.seed_init()

        for i in range( 0, iter ):
            Yh, loss = self.feed_forward()
            self.back_prop()

            if i % 500 == 0:
                print( f"Cost after iteration {i}: {loss}" )
                self.loss.append( loss )

        plt.plot( np.squeeze( self.loss ), color = 'k' )
        plt.ylabel( 'Loss' )
        plt.xlabel( r'Iter ($\times$1000)' )
        plt.title( f"Lr = {str( self.lr )}" )
        plt.show()

        return

    def save_params( self, root = 'nn_params' ):
        np.save( f'{root}.w1.nn.params', self.param["W1"] )
        np.save( f'{root}.w2.nn.params', self.param["W2"] )
        np.save( f'{root}.b1.nn.params', self.param["b1"] )
        np.save( f'{root}.b2.nn.params', self.param["b2"] )

    def load_params( self, root = 'nn_params' ):
        self.param["W1"] = np.load( f'{root}.w1.nn.params.npy' )
        self.param["W2"] = np.load( f'{root}.w2.nn.params.npy' )
        self.param["b1"] = np.load( f'{root}.b1.nn.params.npy' )
        self.param["b2"] = np.load( f'{root}.b2.nn.params.npy' )

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

    training, validation = sys.argv[1], sys.argv[2]
    x, y, xv, yv = read_data( training, validation )

    # Start neural network
    nn = NeuralNet( x, y, 0.006 )
    nn.dims = [len(x), 20, 1]
    nn.threshold = 0.5
    nn.gd( iter = 100000 )

    pred_train = nn.pred_train( x, y, True )
    pred_valid = nn.pred_train( xv, yv, True )

    nn.X, nn.Y = x, y
    target = np.around( np.squeeze( y ), decimals = 0 ).astype(np.int)
    predicted = np.around( np.squeeze( pred_train ), decimals = 0 ).astype(np.int)
    plot_cf( target, predicted, 'Training Set' )

    nn.X, nn.Y = xv, yv
    target = np.around( np.squeeze( yv ), decimals = 0 ).astype(np.int)
    predicted = np.around( np.squeeze( pred_valid ), decimals = 0 ).astype(np.int)
    plot_cf( target, predicted, 'Validation Set' )

    if SAVE:
        nn.save_params( root = 'psr' )
