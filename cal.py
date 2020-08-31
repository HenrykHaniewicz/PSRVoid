#!/usr/local/bin/python3
# New calibrator

"""
Saves as ASCII, all values in Jy/count:         Freq (MHz)         AA          BB         CR         CI
"""

import sys
import os
import numpy as np
import math
import flux
import u
from scipy import interpolate

import matplotlib.pyplot as plt

from pypulse.archive import Archive
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as unit

format = 2

class Cal():

    def __init__( self, file, cont_name, cont_fits_dir, verbose = False ):
        self.file = file
        self.cont_name = cont_name
        self.cont_fits_dir = cont_fits_dir
        self.verbose = verbose
        # Change these to match filename format...
        self.obs_num = self.file[-14:-10]
        self.num = self.file[-9:-5]

        self.ar = Archive( self.file, prepare = True, verbose = self.verbose )
        self.mjd = self.ar.getMJD()
        self.fe = self.ar.getFrontend()


    def get_onoff_data( self, tolerance = 1 ):

        try:
            dat = np.genfromtxt( "../logs/on_off.log", dtype = 'str' )
            on, off, frontend, date, n = dat[:, 0], dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4]
        except OSError:
            on, off, frontend, date, n = [], [], [], [], []
            l = []

            # Get accurate co-ordinates for source
            pos, params = flux.find_source_params_f2( self.cont_name )
            m_coordinates = SkyCoord( f"{pos[0]} {pos[1]}", unit = ( unit.hourangle, unit.degree ) )

            for f in sorted( os.listdir( self.cont_fits_dir ) ):
                continuum_file = os.path.join( self.cont_fits_dir, f )
                try:
                    fe, mjd, cont_ra, cont_dec = u.find_fe_mjd_ra_dec( continuum_file, "CAL" )
                except Exception:
                    continue

                obs_num = continuum_file[-9:-5]

                # Get co-ordinates from FITS file in correct units
                coords = SkyCoord( f"{cont_ra} {cont_dec}", unit = ( unit.hourangle, unit.degree ) )

                # Enter bizarre Henryk mind space (but tell me it doesn't work)
                l.append( { 'MJD' : mjd, 'ON' : None, 'OFF' : None, 'FE' : fe, 'NUM' : obs_num } )

                # Determine if co-ordinates signify ON or OFF source observation. Tolerance in arcminutes.
                if m_coordinates.separation( coords ) <= ( tolerance * unit.arcmin ):
                    mode = 'ON'
                else:
                    mode = 'OFF'

                # Set the filename to the correct flag in the dictionary
                for dict in l:
                    if (dict[ 'MJD' ] == mjd) and (dict[ 'FE' ] == fe) and (dict[ 'NUM' ] == obs_num) and (dict[ mode ] is None):
                        dict[ mode ] = f

            # Delete the excess rows that got created
            for dict in reversed( l ):
                if (dict[ 'ON' ] is None) or (dict[ 'OFF' ] is None):
                    l.remove( dict )

            for d in l:
                on.append( d[ 'ON' ] )
                off.append( d[ 'OFF' ] )
                frontend.append( d[ 'FE' ] )
                date.append( d[ 'MJD' ] )
                n.append( d[ 'NUM' ] )

            if self.verbose:
                print( "Saving to logs/on_off.log" )

            with open( "../logs/on_off.log", "w" ) as f:
                data = np.array( [ on, off, frontend, date, n ] ).T
                np.savetxt( f, data, fmt="%s" )

        # Returns everything as iterables
        return on, off, frontend, date, n

    def closest_continuum2psrcal( self, mjd_tol = 400 ):

        try:
            dat = np.genfromtxt( "../logs/psrcal2continuum.log", dtype = 'str' )
            p, c_on, c_off, m, c_m, fronts = dat[:, 0], dat[:, 1], dat[:, 2], dat[:, 3], dat[:, 4], dat[:, 5]
        except OSError:
            # Then make the table

            on, off, frontend, date, _ = self.get_onoff_data( tolerance = 1 )
            p, c_on, c_off, m, c_m, fronts = [], [], [], [], [], []

            for psr_cal in sorted( os.listdir( os.getcwd() ) ):
                mjd_list = []
                try:
                    psr_fe, psr_mjd = u.find_fe_mjd( psr_cal, "CAL" )
                except OSError:
                    continue

                if (psr_mjd == -1) or (psr_fe == -1):
                    continue

                # Compare psr_cal file with continuum files
                for cont_on, cont_off, cont_fe, cont_mjd in zip( on, off, frontend, date ):
                    if cont_fe == psr_fe:
                        delta_mjd = abs( int(cont_mjd) - psr_mjd )
                        if all( elem > delta_mjd for elem in mjd_list ):
                            mjd_list.append( delta_mjd )
                            if delta_mjd < mjd_tol:
                                continuum_on = str( os.path.join( self.cont_fits_dir, cont_on ) )
                                continuum_off = str( os.path.join( self.cont_fits_dir, cont_off ) )
                                if psr_cal in p:
                                    pind = p.index( psr_cal )
                                    if delta_mjd < abs( int(c_m[pind]) - psr_mjd ):
                                        p[pind] = psr_cal
                                else:
                                    p.append( psr_cal )
                                    m.append( psr_mjd )
                                    c_m.append( cont_mjd )
                                    c_on.append( continuum_on )
                                    c_off.append( continuum_off )
                                    fronts.append( psr_fe )
                            else:
                                continuum_on, continuum_off = None, None

            if self.verbose:
                print( "Saving to logs/psrcal2continuum.log" )

            with open( "../logs/psrcal2continuum.log", "w" ) as f:
                data = np.array( [ p, c_on, c_off, m, c_m, fronts ] ).T
                np.savetxt( f, data, fmt="%s" )

        return p, c_on, c_off, m, c_m, fronts


    def jpc( self, G = 11.0, **kwargs ):

        psr_cal, cont_on, cont_off, psr_mjd, cont_mjd, fe = self.closest_continuum2psrcal()
        freq = self.ar.freq[0]

        inds = [i for i, n in enumerate( psr_mjd ) if int(n) == int(self.mjd)]
        ind = None
        for i in inds:
            if fe[i] == self.fe:
                ind = i


        psr_ar = Archive( psr_cal[ind], prepare = False, verbose = False )
        cont_on_ar = Archive( cont_on[ind], prepare = False, verbose = False )
        cont_off_ar = Archive( cont_off[ind], prepare = False, verbose = False )

        cal_arc_list = []

        np.set_printoptions( threshold = sys.maxsize )
        for ar in [ psr_ar, cont_on_ar, cont_off_ar ]:
            ar.tscrunch()
            fr = ar.freq[0]
            data = ar.getData()
            print(data.shape)
            # The following commented lines are for when certain frequencies are missing
            if (ar is cont_on_ar) or (ar is cont_off_ar):
                # if self.fe == '430':
                #     data = np.delete( data, slice(48, 56), 1 )
                #     fr = np.delete( fr, slice(48, 56), 0 )
                if self.fe == 'lbw':
                    #
                    #data = np.delete( data, slice(0, 64), 1 )
                    #fr = np.delete( fr, slice(0, 64), 0 )
                    #data = np.delete( data, slice(192, 256), 1 )
                    #fr = np.delete( fr, slice(192, 256), 0 )
                    data = np.delete( data, slice(256, 320), 1 )
                    fr = np.delete( fr, slice(256, 320), 0 )
                    #data = np.delete( data, slice(320, 384), 1 )
                    #fr = np.delete( fr, slice(320, 384), 0 )
                #print(fr)
                # if self.fe == '430':
                #     for i in range(8):
                #         data = np.insert( data, 48, np.zeros((2048)), axis = 1 )
                #     fr = np.insert( fr, 48, np.linspace( 455.0, 467.5, 8, endpoint = False ), 0 )
                # elif self.fe == 'lbw':
                #     for i in range(64):
                #         data = np.insert( data, 384, np.zeros((2048)), axis = 1 )
                #     fr = np.insert( fr, 384, np.linspace( 1180.0, 1080.0, 64, endpoint = False ), 0 )

            cal_arc_list.append( { 'ARC' : ar, 'DATA' : data, 'FREQS' : fr, 'S_DUTY' : ar.getValue( 'CAL_PHS' ) , 'DUTY' : ar.getValue( 'CAL_DCYC' ), 'BW' : ar.getBandwidth() } )

        cal_fluxes = flux.get_fluxes( freq/1000, self.cont_name, **kwargs )

        # Sometimes the calibration happens the opposite way, in which case this would be H, L
        L, H = self._prepare_calibration( cal_arc_list )


        F_ON = ( H[1]/L[1] ) - 1
        F_OFF = ( H[2]/L[2] ) - 1


        C0 = cal_fluxes / ( ( 1 / F_ON ) - ( 1 / F_OFF ) )
        T_sys = C0 / F_OFF
        F_cal = ( T_sys * F_OFF ) / G
        F_cal = np.nan_to_num( F_cal, nan = 0.0 )

        # Plots F_cal if first file in series (for checking purposes)
        if self.num == '0001':
           for f in F_cal:
               plt.plot(f)
           plt.show()

        Fa, Fb = interpolate.interp1d( cal_arc_list[1][ 'FREQS' ], F_cal[0], kind = 'cubic', fill_value = 'extrapolate' ), interpolate.interp1d( cal_arc_list[2][ 'FREQS' ], F_cal[1], kind = 'cubic', fill_value = 'extrapolate' )
        Fcr, Fci = interpolate.interp1d( cal_arc_list[1][ 'FREQS' ], F_cal[2], kind = 'cubic', fill_value = 'extrapolate' ), interpolate.interp1d( cal_arc_list[2][ 'FREQS' ], F_cal[3], kind = 'cubic', fill_value = 'extrapolate' )

        aa = Fa( cal_arc_list[0][ 'FREQS' ] ) / ( H[0][0] - L[0][0] )
        bb = Fb( cal_arc_list[0][ 'FREQS' ] ) / ( H[0][1] - L[0][1] )
        cr = Fcr( cal_arc_list[0][ 'FREQS' ] ) / ( H[0][2] - L[0][2] )
        ci = Fci( cal_arc_list[0][ 'FREQS' ] ) / ( H[0][3] - L[0][3] )

        with open( f"../cal/{self.ar.getName()}_{self.fe}_{int(psr_mjd[ind])}_{self.obs_num}_{self.num}.cal", "w" ) as f:
            data = np.array( [ freq, aa, bb, cr, ci ] ).T
            np.savetxt( f, data )

        return freq, aa, bb, cr, ci


    def _prepare_calibration( self, archive_list, r_err = 8 ):

        H = []
        L = []

        for dict in archive_list:
            all_high_means = []
            all_low_means = []

            for pol in dict[ 'DATA' ]:
                high_means = []
                low_means = []

                for i, channel in enumerate( pol ):

                    start_bin = math.floor( len( channel ) * dict[ 'S_DUTY' ] )
                    mid_bin = math.floor( len( channel ) * ( dict[ 'S_DUTY' ] + dict[ 'DUTY' ] ) )
                    end_bin = mid_bin + ( math.floor( len( channel ) * dict[ 'DUTY' ] ) )
                    bin_params = [ start_bin, mid_bin, end_bin ]

                    low_mean = np.mean( channel[ bin_params[0] : bin_params[1] ] )
                    high_mean = np.mean( channel[ bin_params[1] : bin_params[2] ] )

                    low_mean = round( low_mean, r_err )
                    high_mean = round( high_mean, r_err )

                    high_means.append( high_mean )
                    low_means.append( low_mean )

                all_high_means.append( high_means )
                all_low_means.append( low_means )

            H.append( all_high_means )
            L.append( all_low_means )

        H = np.array(H)
        L = np.array(L)

        return H, L




if __name__ == "__main__":

    args = sys.argv[1:]
    c = Cal( args[0], 'B1442', '../../J1829+2456/cont/', True )
    freq, aa, bb, cr, ci = c.jpc( G = 11.0 )
