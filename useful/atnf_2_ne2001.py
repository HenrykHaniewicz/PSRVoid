#!/usr/local/bin/python3
# Gets NE2001 distances from co-ordinates and DMs in the ATNF Catalog
# Manchester, R. N., Hobbs, G. B., Teoh, A. and Hobbs, M., AJ, vol. 129, 1993-2006 (2005)
# Must have NE2001 installed on local machine
# Cordes, J. M. & Lazio, T. J. W. 2002, "NE2001. I. A New Model for the Galactic Distribution of Free Electrons and its Fluctuations"

# If you have a different method of running NE2001, just edit the second command below

NE2001_DIR = '/usr/local/src/NE2001/bin.NE2001'

import sys
from atnf import ATNF
from u import run_cmd

if __name__ == "__main__":

    atnf = ATNF( sys.argv[1:] )
    psrs = atnf.psrs

    for i in range(len( psrs )):
        print( atnf.names[i] )
        cmd = f'./radec2gal.pl {atnf.position[i]}'
        out, err = run_cmd( cmd )
        out_d = out.decode()
        pos_lb = out_d[:-1]

        cmd = f'{NE2001_DIR}/NE2001 {pos_lb} {atnf.dm[i]} 1'
        out, err = run_cmd( cmd )
        out_d = out.decode()
        ne2001_out = out_d.split('\n')[0].split()[10]

        ne2001_out = float(ne2001_out)/1000

        print( f'{ne2001_out:.2f} kpc' )
