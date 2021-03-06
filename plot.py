# Plotting utilities

# Local imports
import physics
import u
from flux import get_flux

# External imports
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

# Hard coded color list - what fun!
color_list = [ 'r', 'g', 'b', 'c', 'y', 'm' ]


def plot_flux( source, frequency, format1 = False, show = False, **kwargs ):

    """
    Plots flux vs frequency for a given source and format. Requires fluxcal.cfg.
    """

    fluxes = []
    for f in frequency:
        flux = get_flux( f, source, format1 )
        fluxes.append( flux )

    if show:
        # Set up figure and axes
        fig = plt.figure()
        ax = fig.add_subplot( 111, facecolor = 'w' )

        ax.set_xlabel( r"Frequency (GHz)" )
        ax.set_ylabel( r"Flux (Jy)" )
        ax.plot( freqs, fluxes, **kwargs )
        plt.show()
        plt.close()
        return 0
    else:
        return fluxes

# Plots NN confusion matrix
def plot_cf( a, b, t, save = None ):
    cf = confusion_matrix( a, b )
    plt.imshow( cf, cmap = plt.cm.Blues, interpolation = 'nearest', aspect = 'auto' )
    plt.colorbar()
    plt.title(t)
    plt.xlabel( 'Predicted' )
    plt.ylabel( 'Actual' )
    tick_marks = np.arange( len( set(a) ) ) # length of classes
    class_labels = [ '0', '1' ]
    plt.xticks( tick_marks, class_labels )
    plt.yticks( tick_marks, class_labels )
    plt.ylim( 1.5, -0.5 )
    thresh = cf.max() / 2.

    for i, j in itertools.product( range( cf.shape[0] ), range( cf.shape[1] ) ):
        plt.text( j, i, format( cf[i,j], 'd' ), horizontalalignment = 'center', verticalalignment = 'center', color = 'white' if cf[i, j] > thresh else 'black')
    plt.tight_layout()
    if save != None:
        plt.savefig( save, format = 'pdf' )
    plt.show()


def histogram_and_curves( array, mean = 0.0, std_dev = 1.0, bins = None, x_lims = None, y_lims = None, x_axis = 'X', y_axis = 'Y', title = 'Title', show = False, filename = None, curve_list = None, labels = None, **kwargs ):

    """
    Histogram plotter for 1 or 2D data. Can compare PDFs in 1D

    Parameters
    ----------
    array      : np.ndarray
        1 or 2D data array
    mean       : int, float, [int, int], [float, float]
        Calculated mean of data
    std_dev    : int, float, [int, int], [float, float]
        Calculated standard deviation of data
    bins       : int
        Number of bins in histogram
    x_lims, y_lims : [int, int], [float, float]
        x and y limits of the plot
    x_axis, y_axis, title : str
        x, y and title names
    show       : bool
        Show plots (default is False)
    filename   : str
        Name of the file to save to (if None, the plot will not be saved)
    curve_list : list of callables
        List of curves to fit to the data as defined by the user
    labels     : [str, str, ...]
        List of legend labels for the curve list
    **kwargs
        kwargs passed to np.hist()

    Returns
    -------
    matplotlib Axes : ax
    """

    color = 'k'
    bgcolor = 'w'
    style = 'stepfilled'

    # Set up figure and axes
    fig = plt.figure( figsize = ( 6, 6 ) )
    ax = fig.add_subplot( 111, facecolor = bgcolor )
    xText = ax.set_xlabel( x_axis )
    yText = ax.set_ylabel( y_axis )
    title = ax.set_title( title )

    # Convert any lists or dicts to numpy arrays
    if not isinstance( array, np.ndarray ):
        array = np.array( array )

    if array.ndim is 1:

        # Number of histogram bins to use (if not supplied by user)
        if bins is None:
            step = ( math.ceil( np.amax( array ) ) - math.floor( np.amin( array ) ) ) / ( 20 * abs( math.ceil( np.amax( array ) ) ) )
            bins = np.arange( math.floor( np.amin( array ) ), math.ceil( np.amax( array ) ), step )

        if not x_lims:
            x_min = mean - ( 4 * std_dev )
            x_max = mean + ( 4 * std_dev )
        else:
            x_min, x_max = x_lims

        # Linespace for curve plotting
        t = np.arange( x_min , x_max, 0.01)

        # Plot the 1D histogram
        n, bins, patches = ax.hist( array, bins = bins, density = True, color = color, histtype = style, linewidth = 2, **kwargs )

        xlim = ax.set_xlim( x_min, x_max )
        ylim = ax.set_ylim( 0, 1.2 * np.amax( n ) )

        # Plot distribution curves
        if curve_list:
            for i, curve in enumerate( curve_list ):

                # Selects color from list
                color_index = i % len( color_list )
                color = color_list[ color_index ]

                # Find the number of fitting arguments in the desired curve
                p0_len = u.get_unique_fitting_parameter_length( curve )

                if not p0_len:
                    p0 = [ mean, std_dev ]
                else:
                    p0 = np.ones( p0_len )


                # Try fitting and plotting. If fit doesn't work, just plot the histogram
                try:
                    try:
                        params = opt.curve_fit( curve, bins[1:], n, p0 = p0 )
                    except RuntimeError:
                        continue

                    line, = ax.plot( t, curve( t, *params[0] ), color = color, linewidth = 2 )
                except TypeError:
                    line, = ax.plot( t, curve( t ), color = color, linewidth = 2 )

            if labels:
                leg_labs = np.zeros( len( labels ) )

                if len( labels ) == len( curve_list ):
                    leg_labs[i] = line.set_label( labels[i] )
                    ax.legend()


        plt.grid( True )

        # Save file
        if filename is not None:
            plt.savefig( filename )
        if show:
            plt.show()
            plt.close()
        else:
            plt.close()

    elif array.ndim is 2 and array.shape[0] is 2:

        # Basically determines whether mean given was [float, float] or float
        if not x_lims:
            try:
                x_min = mean[0] - ( 4 * std_dev[0] )
                x_max = mean[0] + ( 4 * std_dev[0] )
            except TypeError:
                x_min = mean - ( 4 * std_dev )
                x_max = mean + ( 4 * std_dev )
        else:
            x_min, x_max = x_lims

        if not y_lims:
            try:
                y_min = mean[1] - ( 4 * std_dev[1] )
                y_max = mean[1] + ( 4 * std_dev[1] )
            except TypeError:
                y_min = x_min
                y_max = x_max
        else:
            y_min, y_max = y_lims

        # Initialize bin size if not parsed in
        if bins is None:
            step = ( math.ceil( np.amax( array ) ) - math.floor( np.amin( array ) ) ) / ( 100 * abs( math.ceil( np.amax( array ) ) ) )
            bins = [ np.arange( math.floor( np.amin( array[0] ) ), math.ceil( np.amax( array[0] ) ), step ), np.arange( math.floor( np.amin( array[1] ) ), math.ceil( np.amax( array[1] ) ), step ) ]

        # Plot the 2D histogram
        h, x_edge, y_edge, quad_mesh = ax.hist2d( array[0], array[1], bins = bins, **kwargs )

        xlim = ax.set_xlim( x_min, x_max )
        ylim = ax.set_ylim( y_min, y_max )

        if filename is not None:
            plt.savefig( filename )
        if show:
            plt.show()
            plt.close()
        else:
            plt.close()

    elif array.ndim is 2:
        raise ValueError( "Invalid array shape. Number of rows required: 2. (Actual: {})".format( array.shape[0] ) )
    else:
        raise ValueError( "Invalid dimensions. Required: 2. (Actual: {})".format( array.ndim ) )

    return ax


def waterfall( array, ax = None, offset = None, border = 0, labels = True, bins = None, show = True, **kwargs ):

    """
    Waterfall plot of an array. Requires an array of 2 dimensions.
    """

    if array.ndim is 2:
        if offset is None:
            offset = np.max( np.average( array, axis = 0 ) )

        fig = plt.figure( figsize = ( 6, 6 ) )
        bgcolor = 'w'
        ax = fig.add_subplot( 111, facecolor = bgcolor, **kwargs )
        color = 'k'

        if bins is None:
            bins = np.arange( array.shape[1] )


        x_min = 0
        x_max = len( bins ) - 1
        y_min = 0 - offset
        y_max = ( 1 + len( array ) ) * offset
        x_low = x_min - ( x_max - x_min ) * border
        x_high = ( x_max - x_min ) * border + x_max
        y_low = y_min - ( y_max - y_min ) * border
        y_high = ( y_max - y_min ) * border + y_max


        for i in np.arange( len( array ) ):
            ax.plot( array[i][bins] + offset * i, color )

        ax.set_xlim( x_low, x_high )
        ax.set_ylim( y_low, y_high )

        if not labels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        if show:
            plt.show()
        else:
            plt.close()
    else:
        raise ValueError( "Invalid dimensions. Required: 2. (Actual: {})".format( array.ndim ) )

    return ax


def greyscale( array, ax = None, cbar = False, mask = None, show = True, filename = None, setnan = 0.0, **kwargs ):

    """Basic imshow of array"""

    if array.ndim is 2:
        if mask is not None:
            imshow( np.ma.masked_array( array, mask = mask ), ax = ax, **kwargs )
        else:
            imshow( array, ax = ax, **kwargs )
        if cbar:
            plt.colorbar()
        if filename is not None:
            plt.savefig( filename )
        if show:
            plt.show()
        else:
            plt.close()
    else:
        raise ValueError( "Invalid dimensions. Required: 2. (Actual: {})".format( array.ndim ) )

    return ax

# Wrapper for matplotlib imshow
def imshow( x, ax = None, origin = 'lower', interpolation = 'nearest', aspect = 'auto', **kwargs ):
    if ax is not None:
        im = ax.imshow( x, origin = origin, interpolation = interpolation, aspect = aspect, **kwargs )
    else:
        im = plt.imshow( x, origin = origin, interpolation = interpolation, aspect = aspect, **kwargs )
    return im


# (conv function) Plots and shows a vector over a range and maybe a curve or two using matplotlib then frees up memory
def plot_curves( vector, range = None, *curves ):
    plt.plot( vector, color = 'k' )
    if curves and range is not None:
        for i, curve in enumerate( curves ):
            color_index = i % len( color_list )
            color = color_list[ color_index ]
            plt.plot( range, curve, color = color )
    plt.show()
    plt.close()
