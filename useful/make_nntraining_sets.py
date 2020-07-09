#!/usr/local/bin/python3
# Makes training and validation sets for NN RFI exicison

import sys
import numpy as np
from pypulse.archive import Archive
import matplotlib.pyplot as plt

'''
Manual:
    Takes in file (430) or files (lbw)
    Plots each profile, asking the user if that profile is noise or signal
    Rudimentary (I'm assuming the user will get 'most' right, and that's all we need)
    Saves 1 out of 5 of normalized profiles as two pandas lists of nbin and 1 columns
    5th profile saves as two validation pandas lists (of nbin and 1)
    AND SAVE Y AXIS REFLECTION FOR TWICE THE TRAINING SET
    Training file saves as {PSR}_{MJD}_{fe}_{#profs}.training
    Validation file saves as {PSR}_{MJD}_{fe}_{#profs}.validation
Auto:
    Maybe
'''

def load_archive( file, tscrunch = False ):

    ar = Archive( file, verbose = False )
    if tscrunch:
        ar.tscrunch()
        ar.imshow()
    name = ar.getName()
    mjd = int(ar.getMJD())
    fe = ar.getFrontend()
    nbin = ar.getNbin()
    data = ar.getData().reshape( ( ar.getNchan() * ar.getNsubint(), nbin ) )

    return name, mjd, fe, nbin, data


if __name__ == "__main__":

    files = sys.argv[1:]
    i = 0

    for f in files:
        psr, mjd, fe, n, data = load_archive( f, tscrunch = True )
        if f is files[0]:
            d = np.array( data )
        else:
            d = np.vstack(( d, data ))

    with open( f'{psr}_{mjd}_{fe}_{n}.training', 'w' ) as t:
        t.write( f'# Training set for {psr} taken on {mjd} at {fe}\n' )
    with open( f'{psr}_{mjd}_{fe}_{n}.validation', 'w' ) as t:
        t.write( f'# Validation set for {psr} taken on {mjd} at {fe}\n' )

    for profile in d:
        print(i)
        plt.plot( profile, linewidth = 0.3, color = 'k' )
        plt.show( block = False )

        choice = input( "Good profile? (1 is yes, everything else is no): " )
        plt.close()
        try:
            choice = int(choice)
            if choice != 1:
                choice = 0
        except ValueError:
            choice = 0

        # Creates the profile / choice pairs and doubles up with the reciprocal profiles.
        p = np.append( profile, choice )
        inv_p = np.append( -1*profile, choice )
        if (i+1) % 6 == 0:
            with open( f'{psr}_{mjd}_{fe}_{n}.validation', 'a' ) as t:
                np.savetxt( t, p, fmt = '%1.5f ', newline = '' )
                t.write( "\n" )
                np.savetxt( t, inv_p, fmt = '%1.5f ', newline = '' )
                t.write( "\n" )
        else:
            with open( f'{psr}_{mjd}_{fe}_{n}.training', 'a' ) as t:
                np.savetxt( t, p, fmt = '%1.5f ', newline = '' )
                t.write( "\n" )
                np.savetxt( t, inv_p, fmt = '%1.5f ', newline = '' )
                t.write( "\n" )

        i += 1
