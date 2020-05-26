import math
import matplotlib.pyplot as plt
import numpy as np

def addErr( xi, dxi ):
    if len( xi ) == 1:
        return dx
    if len( xi ) != len( dxi ):
        raise ValueError( "Please ensure all values have errors" )
    s = 0
    for val in dxi:
        s += val**2
    return math.sqrt( s )

def multErr( X, xi, dxi ):
    if len( xi ) == 1:
        return dx
    if len( xi ) != len( dxi ):
        raise ValueError( "Please ensure all values have errors" )
    s = 0
    for i, val in enumerate( dxi ):
        s += (val / xi[i])**2
    s = math.sqrt( s )
    s *= abs( X )
    return s

def convErr( x, dx, n ):
    dX = math.sqrt( n ) * abs( x**n ) * ( dx / x )
    return dX

def hypotenuse_error( xi, dxi ):
    # Propagates the uncertainty of a hypotenuse of n independent variables
    hyp, err = 0, 0
    for x, dx in zip(xi, dxi):
        hyp += x**2
        err += 2*x*dx

    hyp = math.sqrt( hyp )
    err *= 0.5*(1/hyp)
    return err


def calculate_errors_from_hist( data, bins, interval, show = False, **kwargs ):
    percent_goal = ( (100 - interval) / 200 ) * sum( data )
    i = 0
    for n, v in enumerate( data ):
        i += v
        if i > percent_goal:
            start = bins[n]
            if show:
                plt.axvline( x = bins[n], **kwargs )
            break
    i = 0
    for n, v in reversed( list( enumerate( data ) ) ):
        i += v
        if i > percent_goal:
            end = bins[n]
            if show:
                plt.axvline( x = bins[n], **kwargs )
            break

    median = np.median( data )
    high = abs( end - median )
    low = abs( median - start )

    return low, high
