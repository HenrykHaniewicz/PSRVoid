#!/usr/local/bin/python3
# Makes training and validation sets for NN RFI exicison

import sys
import numpy as np
from sklearn.model_selection import train_test_split
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
    Training file saves as {PSR}_{MJD}_{fe}_{#bins}.training
    Validation file saves as {PSR}_{MJD}_{fe}_{#bins}.validation
Auto:
    Maybe
'''

def load_archive( file, tscrunch = False ):

    ar = Archive( file, verbose = False )
    if tscrunch:
        ar.tscrunch(nsubint = 4)
        #ar.imshow()
    name = ar.getName()
    mjd = int(ar.getMJD())
    fe = ar.getFrontend()
    nbin = ar.getNbin()
    data = ar.getData().reshape( ( ar.getNchan() * ar.getNsubint(), nbin ) )

    return name, mjd, fe, nbin, data


if __name__ == "__main__":

    TEST_SIZE = 0.2

    files = sys.argv[1:]
    i, j = 1, 1

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

    a = 0
    b = -1

    ps_0 = np.zeros(2049)[np.newaxis, :]
    ps_1 = np.zeros(2049)[np.newaxis, :]

    for profile in d:
        print(i, j)
        plt.plot( profile, linewidth = 0.3, color = 'k' )
        plt.show( block = False )

        choice = input( "Good profile? (1 is yes, everything else is no): " )
        plt.close()
        try:
            choice = int(choice)
            if choice != 1:
                choice = 0
        except ValueError:
            choice = -1

        if choice != -1:
            # Creates the profile / choice pairs and doubles up with the reciprocal profiles.
            p = np.append( profile, choice )
            #inv_p = np.append( -1*profile, choice )
            if choice == 0:
                ps_0 = np.append( ps_0, p[np.newaxis, :], axis = 0 )
            else:
                ps_1 = np.append( ps_1, p[np.newaxis, :], axis = 0 )

            j += 1

        i += 1

    ps_0, ps_1 = np.delete( ps_0, 0, 0 ), np.delete( ps_1, 0, 0 )

    # Sort into training / validation sets
    train, validation = train_test_split( ps_0, test_size = TEST_SIZE )
    ones_t, ones_v = train_test_split( ps_1, test_size = TEST_SIZE )
    train, validation = np.append( train, ones_t, axis = 0 ), np.append( validation, ones_v, axis = 0 )

    np.random.shuffle( train ), np.random.shuffle( validation )

    for k in train:
        with open( f'{psr}_{mjd}_{fe}_{n}.training', 'a' ) as t:
            np.savetxt( t, k, fmt = '%1.5f ', newline = '' )
            t.write( "\n" )
            #np.savetxt( t, inv_p, fmt = '%1.5f ', newline = '' )
            #t.write( "\n" )

    for k in validation:
        with open( f'{psr}_{mjd}_{fe}_{n}.validation', 'a' ) as t:
            np.savetxt( t, k, fmt = '%1.5f ', newline = '' )
            t.write( "\n" )
            #np.savetxt( t, inv_p, fmt = '%1.5f ', newline = '' )
            #t.write( "\n" )
