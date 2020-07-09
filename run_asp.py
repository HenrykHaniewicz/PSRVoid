#!/usr/local/bin/python3
# Runs ASPFitsReader (c) Robert D. Ferdman within Python 3
# Henryk T. Haniewicz, 2020

import sys
import os
import u
from pushsafer import init, Client

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

    for i in range(1, 3):
        #os.system( f"ASPFitsReader -parfile ../tempo1_ddgr.par -dedisp 0 -nsubs 1 -adddumps 195 -infile puppi_{i}_J1829+2456_{j}_0001.fits -outroot '../LC/J1829+2456_{i}_430_{j}_0001' -calfile Cal/J1829+2456_430_{i}_{j}_0001.cal -zapfile Zap/zap_puppi_{i}_J1829+2456_{j}_0001.fits.ascii" )
        os.system( f"ASPFitsReader -parfile ../tempo1_ddgr.par -dedisp 0 -nsubs 1 -adddumps 64 -infile puppi_58402_J1829+2456_0390_000{i}.fits -outroot '../LC/J1829+2456_58402_lbw_0390_000{i}' -calfile Cal/J1829+2456_lbw_58402_0390_000{i}.cal -zapfile Zap/zap_puppi_58402_J1829+2456_0390_000{i}.fits.ascii" )

    exit()
#ASPFitsReader -parfile ../tempo1_ddgr.par -dedisp 0 -nsubs 1 -adddumps 195 -infile puppi_58402_J1829+2456_0486_0001.fits -outroot '../LC/J1829+2456_58402_430_0486_0001' -calfile Cal/J1829+2456_430_58402_0486_0001.cal -zapfile Zap/zap_puppi_58402_J1829+2456_0486_0001.fits.ascii
'''
ASPToa -tempo2 -template ../J1829+2456_lbw_nchan1_template_7gaussianfit.ascii -parfile ../tempo1_ddgr.par -toafile J1829+2456_lbw_58402.toa -t2flags "-f PUPPI_1400_ASP" -infile J1829+2456_58402_lbw_0390_0001.stokes.fits J1829+2456_58402_lbw_0390_0002.stokes.fits J1829+2456_58402_lbw_0390_0003.stokes.fits J1829+2456_58402_lbw_0390_0004.stokes.fits

'''

'''
    if len( sys.argv ) > 4:
        run( sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] )
    elif len( sys.argv ) > 3:
        run( sys.argv[1], sys.argv[2], sys.argv[3] )
    else:
        run( sys.argv[1], sys.argv[2] )
'''
