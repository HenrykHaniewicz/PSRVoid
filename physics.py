# math and physics utilities
# Henryk T. Haniewicz, 2018

# Imports
import math
import scipy.stats
import numpy as np
from scipy.stats import rv_continuous

#import u

# PDF classes
class test_dist( rv_continuous ):

    def _pdf( x, a, b, c ):
        return np.sqrt(a) * ( np.exp(-(b*x)**2 / c ) / np.sqrt(2.0 * np.pi) )

class FFT_dist( rv_continuous ):

    def _pdf( x, b, a, k ):
        return (b/(np.sqrt(1 + a*((k-x)**2))))

# Functions

def dim( a ):
    if not type( a ) == list:
        return []
    return [len( a )] + dim( a[0] )

def multi_norm( x, *args ):
    ret = None
    n_gauss = len( args )//3

    if len( args ) % 3 != 0:
        print( "Args supplied must be a multiple of 3 of form: mu, sig, amp" )
    else:
        ret = 0
        for i in np.arange( 0, 3*n_gauss, 3 ):
            ret += args[i + 2]*scipy.stats.norm.pdf( x, loc = args[i], scale = args[i + 1] )
    return ret


def norm( x, m, s, k ):
    ret = k*scipy.stats.norm.pdf( x, loc = m, scale = s )
    return ret


def reduced_mass( m1, m2 ):
    return (m1*m2)/(m1 + m2)

def arr_reduced_mass( arr_m1, arr_m2 ):
    if len( arr_m2 ) != len( arr_m2 ):
        raise ValueError( "Masses must have a 1:1 ratio (arrays must be of same length)" )
    is_np = (type( arr_m1 ) == np.ndarray)
    if is_np:
        rm = np.array([])
        for i, elem in enumerate( np.arange( len( arr_m1 ) ) ):
            rm = np.append( rm, reduced_mass( arr_m1[i], arr_m2[i] ) )
    else:
        rm = []
        for i, elem in enumerate( np.arange( len( arr_m1 ) ) ):
            rm.append( reduced_mass( arr_m1[i], arr_m2[i] ) )
    return rm

def sigmoid( Z ):
    return 1/( 1 + np.exp( -Z ) )

def relu( Z ):
    return np.maximum( 0, Z )

def lrelu( Z, a = 0.1 ):
    return np.maximum( a*Z, Z )

def softmax( Z ):
    exp = np.exp( Z - np.max(Z) )
    s = np.sum( exp )
    return np.divide( exp, s )

def Swish( Z, b = 1 ):
    return ( Z * sigmoid( b*Z ) )

def drelu2( dZ, Z ):
    dZ[Z <= 0] = 0
    return dZ

def drelu( Z ):
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z

def dlrelu( Z, a = 0.1 ):
    Z[Z < 0] = a
    Z[Z > 0] = 1
    return Z


def dsigmoid( Z ):
    s = 1/( 1 + np.exp( -Z ) )
    dZ = s * ( 1 - s )
    return dZ

def dtanh( Z ):
    return ( 1 - (np.tanh(Z))**2 )

def dsoftmax( Z ):
    S = softmax( Z ).T
    S_vector = np.reshape( S, (S.shape[0], 1) )
    S_matrix = np.tile( S_vector, S.shape[0] )
    dZ = np.diag(S_matrix) - ( S_matrix * np.transpose(S_matrix) )
    return dZ

def dSwish( Z, b = 1 ):
    A = Swish( Z, b )
    return ( A + ( sigmoid( Z ) * ( 1 - A ) ) )


def rms( array ):
    return np.sqrt( np.mean( np.power( array, 2 ) ) )

def calculate_rms_matrix( array, mask = None, mask_output = False ):

    if mask is None:
        m = np.zeros( array.shape[-1] )
    else:
        m = mask

    if array.ndim == 1:
        r = rms( array[m == 0] )
    elif array.ndim == 2:
        r = []
        for i, prof in enumerate( array ):
            r.append( rms( prof[m == 0] ) )
            #u.display_status( i+1, len( array ) )
    elif array.ndim == 3:
        r = []
        i = 1
        for subint in array:
            sub = []
            for prof in subint :
                sub.append( rms( prof[m == 0] ) )
                #u.display_status( i, len( array ) * len( subint ) )
                i += 1
            r.append( sub )

    if mask_output:
        r = np.ma.array( r, mask = np.isnan( r ) )

    return r


# Taken from scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth( x, window_len = 11, window = 'hanning' ):

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y


def normalizeToMax( array ):

    '''
    Divides all elements in an array by max value in that column
    '''

    return ( array / ( np.max( array, 0 ) + np.spacing( 0 ) ) )


def chauvenet( array, median = 0, std_dev = 1, threshold = 3.0 ):

    '''
    Returns a boolean array of the same shape as the input array based on the
    Chauvenet outlier criterion.
    Default values for the mean and stddev are for a normalized Gaussian but
    it is more useful to use values calculated from your array.
    '''

    absDiff = abs( array - median )

    # Output of 1 means bad channel
    return absDiff > ( threshold * std_dev )


def DMAD( vector, threshold = 3.5 ):

    '''
    Returns a boolean array comparing the Modified Z-Score (MZS) to the given threshold factor.
    Only works with 1D arrays (vectors) but can be iterated over for multiple distributions.
    A return of True implies an outlying data point.
    '''

    if vector.ndim != 1:
        raise ValueError( "Input must be a 1D vector." )

    # Calculate the overall median (allows for masked vectors)
    m = np.ma.median( vector )

    # Calculate the absolute deviation from the true median
    abs_dev = np.abs( vector - m )

    # Calculate the median absolute deviation for both the left and right splits
    left_MAD = np.ma.median( abs_dev[vector <= m] )
    right_MAD = np.ma.median( abs_dev[vector >= m] )

    vector_MAD = left_MAD * np.ones( len( vector ) )
    vector_MAD[vector > m] = right_MAD

    # Calculate the modified Z score
    MZS = 0.6745 * abs_dev / vector_MAD

    # If the value of the vector equals the median, set the MZS to 0
    MZS[vector == m] = 0

    # Return true if the MZS is greater than the threshold
    return MZS > threshold


# Time handlers
def minutes_to_seconds( minutes, seconds ):
    return ( minutes * 60 ) + seconds

def hours_to_seconds( hours, minutes, seconds ):
    return ( hours * 3600 ) + minutes_to_seconds( minutes, seconds )

def days_to_seconds( days, hours, minutes, seconds ):
    return ( days * 86400 ) + hours_to_seconds( hours, minutes, seconds )

def seconds_to_minutes( seconds, format = False ):
    mins = math.floor( seconds / 60 )
    secs = seconds % 60
    if format:
        output = '{0}m\ {1:.2f}s'.format( mins, secs )
    else:
        output = mins, secs

    return output

def seconds_to_hours( seconds, format = False ):
    hours = math.floor( seconds / 3600 )
    remainder = seconds % 3600
    mins, secs = seconds_to_minutes( remainder )
    if format:
        output = '{0}h\ {1}m\ {2:.2f}s'.format( hours, mins, secs )
    else:
        output = hours, mins, secs

    return output

def seconds_to_days( seconds, format = False ):
    days = math.floor( seconds / 86400 )
    remainder = seconds % 86400
    hours, mins, secs = seconds_to_hours( remainder )
    if format:
        output = '{0}d\ {1}h\ {2}m\ {3:.2f}s'.format( days, hours, mins, secs )
    else:
        output = days, hours, mins, secs

    return output
