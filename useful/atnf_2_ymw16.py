#!/usr/local/bin/python3
# Gets YMW16 distances from co-ordinates and DMs in the ATNF Catalog
# Manchester, R. N., Hobbs, G. B., Teoh, A. and Hobbs, M., AJ, vol. 129, 1993-2006 (2005)
# Must have YMW16 installed on local machine
# Yao, Manchester and Wang, Astrophysical Journal, vol. 835 (2017)

YMW_DIR = '/usr/local/src/ymw16_v1.3.2'

import sys
from atnf import ATNF
from u import run_cmd

if __name__ == "__main__":

    atnf = ATNF( sys.argv[1:] )
    psrs = atnf.psrs

    for i in range(len( psrs )):
        print( atnf.names[i] )
        cmd = f'radec2gal.pl {atnf.position[i]}'
        out, err = run_cmd( cmd )
        out_d = out.decode()
        pos_lb = out_d[:-1]

        cmd = f'{YMW_DIR}/ymw16 -d {YMW_DIR} Gal {pos_lb} {atnf.dm[i]} 1'
        out, err = run_cmd( cmd )
        out_d = out.decode()
        ymw_out = out_d.split('\n')[0].split()[10]

        ymw_out = float(ymw_out)/1000

        print( f'{ymw_out:.2f} kpc' )
