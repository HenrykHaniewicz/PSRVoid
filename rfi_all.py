#!/usr/local/bin/python3
# New directory excisor

import sys
import os
from scipy.stats import norm

from rfi import Zap
import u


def zap_it( template, frontend, method = None, **kwargs ):

    #files = os.listdir( os.getcwd() )
    files = [ 'PUPPI_J1851+00_58211_37355_0002.fits',
'PUPPI_J1851+00_58211_37355_0003.fits',
'PUPPI_J1851+00_58211_37355_0004.fits',
'PUPPI_J1851+00_58218_35153_0001.fits',
'PUPPI_J1851+00_58218_35153_0002.fits',
'PUPPI_J1851+00_58218_35153_0003.fits',
'PUPPI_J1851+00_58238_31863_0001.fits',
'PUPPI_J1851+00_58238_31863_0002.fits',
'PUPPI_J1851+00_58238_31863_0003.fits',
'PUPPI_J1851+00_58259_24796_0001.fits',
'PUPPI_J1851+00_58259_24796_0002.fits',
'PUPPI_J1851+00_58320_10632_0001.fits',
'PUPPI_J1851+00_58320_10632_0002.fits',
'PUPPI_J1851+00_58327_10525_0001.fits',
'PUPPI_J1851+00_58327_10525_0002.fits',
'PUPPI_J1851+00_58327_10525_0003.fits',
'PUPPI_J1851+00_58356_01933_0001.fits',
'PUPPI_J1851+00_58356_01933_0002.fits',
'PUPPI_J1851+00_58356_01933_0003.fits',
'PUPPI_J1851+00_58360_02617_0001.fits',
'PUPPI_J1851+00_58360_02617_0002.fits',
'PUPPI_J1851+00_58360_02617_0003.fits' ]

    for i, file in enumerate( sorted( files ) ):
        if u.find_frontend( file ) != frontend:
            continue
        if method == None:
            met = 'chauvenet'

        print( f"{file}" )
        try:
            z = Zap( file, template, met, show = True, curve_list = [norm.pdf], x_lims = [0, 10], **kwargs ) # , show = True, curve_list = [norm.pdf], x_lims = [0, 20], **kwargs
            z.plot_mask()
            z.save( outroot = f"../Zap/{file.split('.')[0]}", ext = ".zap" )
        except ValueError:
            pass
        print( (100*i)/len( files ), end = '\r' )

    print( f"Completed RFI exicison" )

    return 0


if __name__ == "__main__":
    print( f"Directory: {os.getcwd()}" )
    if len( sys.argv ) > 3:
        zap_it( sys.argv[1], sys.argv[2], sys.argv[3] )
    else:
        zap_it( sys.argv[1], sys.argv[2] )
