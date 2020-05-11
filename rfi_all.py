#!/usr/local/bin/python3
# New directory excisor

import sys
import os
from scipy.stats import norm

from rfi import Zap
import u


def zap_it( template, frontend, method = None, **kwargs ):

    for file in sorted( os.listdir( os.getcwd() ) ):
        if u.find_frontend( file ) != frontend:
            continue
        if method == None:
            met = 'chauvenet'

        print( f"{file}" )
        z = Zap( file, template, met, **kwargs ) # , show = True, curve_list = [norm.pdf], x_lims = [0, 20], **kwargs
        z.save( outroot = f"Zap/zap_{file}" )

    print( f"Completed RFI exicison" )

    return 0


if __name__ == "__main__":
    print( f"Directory: {os.getcwd()}" )
    if len( sys.argv ) > 3:
        zap_it( sys.argv[1], sys.argv[2], sys.argv[3] )
    else:
        zap_it( sys.argv[1], sys.argv[2] )
