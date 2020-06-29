#!/usr/local/bin/python3
# Runs ASPFitsReader (c) Robert D. Ferdman within Python 3
# Henryk T. Haniewicz, 2020

import sys
import os
import u

def run( parfile, frontend, fscrunch_fac = 1, adddumps = None ):
    for file in sorted( os.listdir( os.getcwd() ) ):
        fe, nsub, nchan, mjd = u.find_fe_nsub_nchan_mjd( file )
        if fe != frontend:
            continue
        if adddumps:
            nsub = adddumps
        num = file[-9:-5]
        os.system( f"ASPFitsReader -infile {file} -outroot 'NF/J1829+2456_{mjd}_{fe}_PUPPI_{num}' -nsubs {fscrunch_fac} -adddumps {nsub} -dedisp 0 -parfile {parfile} -zapfile Zap/zap_{file}.nn.ascii" )
    return 0

if __name__ == "__main__":
    if len( sys.argv ) > 4:
        run( sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] )
    elif len( sys.argv ) > 3:
        run( sys.argv[1], sys.argv[2], sys.argv[3] )
    else:
        run( sys.argv[1], sys.argv[2] )
