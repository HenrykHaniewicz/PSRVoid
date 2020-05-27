#!/usr/local/bin/python3
# Gets YMW16 distances from co-ordinates and DMs in the ATNF Catalog
# Manchester, R. N., Hobbs, G. B., Teoh, A. and Hobbs, M., AJ, vol. 129, 1993-2006 (2005)
# Must have YMW16 installed on local machine
# Yao, Manchester and Wang, Astrophysical Journal, vol. 835 (2017)

YMW_DIR = '/usr/local/src/ymw16_v1.3.2'

import sys
import atnf
from u import run_cmd

if __name__ == "__main__":

    psrs = sys.argv[1:]
    names, pos, dm = atnf.get_names_ra_dec_dm( psrs )

    for i in range(len(psrs)):
        print(names[i])
        cmd = f'radec2gal.pl {pos[i]}'
        out, err = run_cmd( cmd )
        out_d = out.decode()
        pos_lb = out_d[:-1]

        cmd = f'{YMW_DIR}/ymw16 -d {YMW_DIR} Gal {pos_lb} {dm[i]} 1'
        out, err = run_cmd( cmd )
        out_d = out.decode()
        ymw_out = out_d.split('\n')[0].split()[10]

        ymw_out = float(ymw_out)/1000

        print( f'{ymw_out:.2f} kpc' )
