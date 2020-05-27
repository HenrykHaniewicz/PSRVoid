# utilities

# Imports
import sys
import os
import subprocess
import numpy as np
from astropy.io import fits
import inspect
from collections import namedtuple
import matplotlib.pyplot as plt
import scipy.stats as spyst

from pypulse.singlepulse import SinglePulse
from physics import calculate_rms_matrix

def run_cmd( cmd ):
    proc = subprocess.Popen( cmd, stdout = subprocess.PIPE, shell = True )
    (out, err) = proc.communicate()

    return out, err

def get_data_from_asc( asc_file ):
    data = np.loadtxt( asc_file )
    x = data[:, 0]
    y = data[:, 1]
    return x, y

def rms_arr_properties( array, mask, out_tol = 1.5 ):

    '''
    Returns the RMS array, a linearized RMS array, the mean and standard deviation
    '''

    # Return the array of RMS values for each profile
    r = calculate_rms_matrix( array, mask, True )
    l_shape = 1
    for s in array.shape[:-1]:
        l_shape *= s

    # Reshape RMS array to be linear and store in a new RMS array
    l = np.reshape( r, l_shape )

    # Mean and standard deviation
    m, excess = np.nanmedian( l ), out_tol * spyst.iqr( l, nan_policy = 'omit' )
    outlier_bounds = [ 0, np.nanpercentile( l, 75 ) ]
    outlier_bounds = [ outlier_bounds[0], outlier_bounds[1] + excess ]
    array_to_do_calculations = l[ (l >= outlier_bounds[0]) & (l <= outlier_bounds[1]) ]
    mu, s = np.nanmean( array_to_do_calculations ), np.nanstd( array_to_do_calculations )

    return r, l, mu, s

# Delete and flush the current line on the console
def restart_line():
     sys.stdout.write( '\r' )
     sys.stdout.flush()

# Simple progress bar
def display_status( iteration, MAX_ITER ):
    restart_line()

    sys.stdout.write('{0:<10d}[{1:>3d}%]'.format( iteration, int( 100 * float( iteration )/float( MAX_ITER ) ) ) )
    sys.stdout.flush()

# Checks if two arrays (or lists) of the same shape are equivalent element-wise
def is_similar_array( array1, array2, tolerance = 1e-7 ):

    if not isinstance( array1, np.ndarray ):
        array1 = np.array( array1 )
    if not isinstance( array2, np.ndarray ):
        array2 = np.array( array2 )

    if array1.shape != array2.shape:
        raise ValueError( "Both arrays must have the same shape. Shape of Array 1: {}. Shape of Array 2: {}".format( array1.shape, array2.shape ) )

    if isinstance( tolerance, np.ndarray ):
        if tolerance.shape != array1.shape:
            raise ValueError( "If tolerance is an array, it must have the same shape as the inputs. Shape of tolerance array: {}. Shape of inputs: {}".format( tolerance.shape, array1.shape ) )
    elif isinstance( tolerance, list ):
        tolerance = np.array( tolerance )
        if tolerance.shape != array1.shape:
            raise ValueError( "If tolerance is an array, it must have the same shape as the inputs. Shape of tolerance array: {}. Shape of inputs: {}".format( tolerance.shape, array1.shape ) )

    diff = abs( np.subtract( array1, array2 ) )

    return diff < tolerance


def addExtension( file, ext, save = False, overwrite = False ):

    '''
    Add any desired extension to a file that doesn't have one.
    If the file does, that extension will be used instead unless overwrite is
    checked.
    '''

    if not isinstance( ext, str ):
        raise TypeError( "Extension must be a string" )

    # Split the filename up into a root and extension
    root, end = os.path.splitext( file )

    # If file extension does not exist, add the extension
    if (not end) or overwrite:
        end = '.' + ext
        fileout = root + end
    else:
        fileout = file

    # Rename the file in os if enabled
    if save:
        os.rename( file, fileout )

    return fileout

def getargspec_no_self( func ):
    """
    EDITED FROM SCIPY

    inspect.getargspec replacement using inspect.signature.
    inspect.getargspec is deprecated in python 3. This is a replacement
    based on the (new in python 3.3) `inspect.signature`.
    Parameters
    ----------
    func : callable
        A callable to inspect
    Returns
    -------
    argspec : ArgSpec(args, varargs, varkw, defaults)
        This is similar to the result of inspect.getargspec(func) under
        python 2.x.
        NOTE: if the first argument of `func` is self, it is *not*, I repeat
        *not* included in argspec.args.
        This is done for consistency between inspect.getargspec() under
        python 2.x, and inspect.signature() under python 3.x.
    """

    ArgSpec = namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])

    sig = inspect.signature(func)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    varkw = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    varkw = varkw[0] if varkw else None
    defaults = [
        p.default for p in sig.parameters.values()
        if (p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
           p.default is not p.empty)
    ] or None
    return ArgSpec(args, varargs, varkw, defaults)


def get_unique_fitting_parameter_length( func ):

    argspec = getargspec_no_self( func )

    if len( argspec[0] ) is 0:
        raise ValueError( "Length of ArgSpec must not be 0" )

    has_self = False
    count = 0

    for i, arg in enumerate( argspec[0] ):
        if arg is 'self':
            has_self = True
            continue
        if has_self and i is 1:
            continue
        if not has_self and i is 0:
            continue

        count += 1

    return count

# Takes center frequency, bandwidth and a number of frequency channels. Spits out frequencys centered uniformly about the center.
def chan_to_freq( ctr_freq, bandwidth, nchan ):

    start_freq = ctr_freq - ( bandwidth / 2 )
    end_freq = ctr_freq + ( bandwidth / 2 )
    freq = np.linspace( start_freq, end_freq, num = nchan )
    return freq

# counts files for directories
def count_files( search_string, *dirs, verbose = True ):
    total = []
    final_count = 0
    for d in dirs:
        count = 0
        f_list = os.listdir( d )
        total_count = len( f_list )
        for f in f_list:
            if search_string in f:
                count += 1
        total.append( [ d, total_count, count ] )
        final_count += count
    if verbose:
        for e in total:
                print( f"Directory: {e[0]}\tTotal files: {e[1]}\tFiles found containing '{search_string}': {e[2]}" )
        if not len( dirs ) == 1:
            print( f"Total files found containing '{search_string}': {final_count}" )
    return total, final_count

############# FIND FUNCTIONS ###############

def find_frontend( file, obs = "PSR" ):
    try:
        hdul = fits.open( file )
    except OSError:
        return -1

    if hdul[0].header[ 'OBS_MODE' ] != obs:
        hdul.close()
        return -1
    try:
        fe = hdul[0].header[ 'FRONTEND' ]
    except KeyError:
        hdul.close()
        return -1
    hdul.close()

    return fe

def find_fe_mjd( file, obs = "PSR" ):
    try:
        hdul = fits.open( file )
    except OSError:
        return -1, -1
    if hdul[0].header[ 'OBS_MODE' ] != obs:
        hdul.close()
        return -1, -1

    try:
        fe = hdul[0].header[ 'FRONTEND' ]
        mjd = hdul[0].header[ 'STT_IMJD' ]
    except KeyError:
        hdul.close()
        return -1, -1
    hdul.close()

    return fe, mjd

def find_fe_mjd_ra_dec( file, obs = "PSR" ):
    try:
        hdul = fits.open( file )
    except OSError:
        return -1, -1, -1, -1
    if hdul[0].header[ 'OBS_MODE' ] != obs:
        hdul.close()
        return -1, -1, -1, -1

    try:
        fe = hdul[0].header[ 'FRONTEND' ]
        mjd = hdul[0].header[ 'STT_IMJD' ]
        ra = hdul[0].header[ 'RA' ]
        dec = hdul[0].header[ 'DEC' ]
    except KeyError:
        hdul.close()
        return -1, -1, -1, -1
    hdul.close()

    return fe, mjd, ra, dec

def find_fe_nsub_nchan_mjd( file, obs = "PSR" ):
    try:
        hdul = fits.open( file )
    except OSError:
        return -1, -1, -1, -1
    if hdul[0].header[ 'OBS_MODE' ] != obs:
        hdul.close()
        return -1, -1, -1, -1

    try:
        fe = hdul[0].header[ 'FRONTEND' ]
        nsub = hdul[-1].header[ 'NAXIS2' ]
        nchan = hdul[0].header[ 'OBSNCHAN' ]
        mjd = hdul[0].header[ 'STT_IMJD' ]
    except KeyError:
        hdul.close()
        return -1, -1, -1, -1
    hdul.close()

    return fe, nsub, nchan, mjd

def get_1D_OPW_mask( v, **kwargs ):

    if v.ndim is not 1:
        raise ValueError( "Input data must be 1 dimensional to create an OPW mask." )

    mask = []
    sp_dat = SinglePulse( v, **kwargs )

    while len( mask ) < len( v ):
        if len( mask ) in sp_dat.opw:
            mask.append( False )
        else:
            mask.append( True )

    mask = np.asarray( mask )
    return mask
