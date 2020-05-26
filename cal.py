#!/usr/local/bin/python3
# New calibrator

"""
Saves as ASCII, all values in Jy/count:         Freq (MHz)         AA          BB         CR         CI

STILL UNDER CONSTRUCTION
"""

import sys
import os
import numpy as np
import flux
import u

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
        self.ar = Archive( file, prepare = False, verbose = False )
        self.mjd = self.ar.getMJD()
        self.fe = self.ar.getFrontend()
        #self.ar.dedisperse( wcfreq = self.ar.wcfreq )
        #self.ar.center()
        #self.jy_per_count = self.jpc( **kwargs )


    def get_onoff_data( self, tolerance = 1 ):

        try:
            dat = np.genfromtxt( "logs/on_off.log", dtype = 'str' )
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

            with open( "logs/on_off.log", "w" ) as f:
                data = np.array( [ on, off, frontend, date, n ] ).T
                np.savetxt( f, data, fmt="%s" )

        # Returns everything as iterables
        return on, off, frontend, date, n

    def closest_continuum2psrcal( self, mjd_tol = 50 ):

        try:
            dat = np.genfromtxt( "logs/psrcal2continuum.log", dtype = 'str' )
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

            with open( "logs/psrcal2continuum.log", "w" ) as f:
                data = np.array( [ p, c_on, c_off, m, c_m, fronts ] ).T
                np.savetxt( f, data, fmt="%s" )

        return p, c_on, c_off, m, c_m, fronts


    def jpc( self, G = 11.0, **kwargs ):

        psr_cal, cont_on, cont_off, psr_mjd, cont_mjd, fe = self.closest_continuum2psrcal()
        freq = self.ar.freq

        inds = [i for i, n in enumerate( psr_mjd ) if int(n) == int(self.mjd)]
        ind = None
        for i in inds:
            if fe[i] == self.fe:
                ind = i

        psr_ar = Archive( psr_cal[ind], prepare = False, verbose = False )
        cont_on_ar = Archive( cont_on[ind], prepare = False, verbose = False )
        cont_off_ar = Archive( cont_off[ind], prepare = False, verbose = False )

        for ar in [ psr_ar, cont_on_ar, cont_off_ar ]:
            ar.tscrunch()
            #l = { 'ARC' : arc, 'DATA' : [ A, B ], 'FREQS' : freq[i], 'S_DUTY' : arc.getValue( 'CAL_PHS' ) , 'DUTY' : arc.getValue( 'CAL_DCYC' ), 'BW' : arc.getBandwidth() }

        exit()

        cal_fluxes = flux.get_fluxes( freq/1000, self.cont_name, **kwargs )
        cont_on_level = 0

        H, L, T0 = self._prepare_calibration( aabb_list )
        F_ON = ( H[1]/L[1] ) - 1
        F_OFF = ( H[2]/L[2] ) - 1

        C0 = T0[1:] / ( ( 1 / F_ON ) - ( 1 / F_OFF ) )
        T_sys = C0 / F_OFF
        F_cal = ( T_sys * F_OFF ) / G


        Fa, Fb = interp1d( aabb_list[1][ 'FREQS' ], F_cal[0][0], kind='cubic', fill_value = 'extrapolate' ), interp1d( aabb_list[2][ 'FREQS' ], F_cal[0][1], kind='cubic', fill_value = 'extrapolate' )

        conversion_factor = [ np.array(Fa( aabb_list[0][ 'FREQS' ] ) / ( H[0][0] - L[0][0] )), np.array( Fb( aabb_list[0][ 'FREQS' ] ) / ( H[0][1] - L[0][1] ) ) ]
        conversion_factor = np.array( conversion_factor )

        with open( "cal_factors.ascii", "w" ) as f:
            data = np.array( [ freq, aa, bb, cr, ci ] ).T
            np.savetxt( f, data )

        return freq, aa, bb, cr, ci


    def _prepare_calibration( self, archive_list, r_err = 8 ):

        H = []
        L = []
        T0 = []

        for dict in archive_list:
            all_high_means = []
            all_low_means = []
            T0_pol = []

            for pol in dict[ 'DATA' ]:
                high_means = []
                low_means = []
                T0_chan = []

                for i, channel in enumerate( pol ):

                    flux = getFlux( float( dict[ 'FREQS' ][i]/1000 ), self.cont_name, False )

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
                    T0_chan.append( flux )

                all_high_means.append( high_means )
                all_low_means.append( low_means )
                T0_pol.append( T0_chan )

            H.append( all_high_means )
            L.append( all_low_means )
            T0.append( T0_pol )

        H = np.array(H)
        L = np.array(L)
        T0 = np.array(T0)

        return H, L, T0


    def calibrate( self ):

        """
        Master calibration method
        """

        conv_file = "{}_{}_fluxcalibration_conversion_factors.pkl".format( self.psr_name, self.cont_name )
        cal_mjd_file = "{}_{}_fluxcalibration_cal_mjds.pkl".format( self.psr_name, self.cont_name )
        conv_abs_path, cal_abs_path = os.path.join( self.pkl_dir, 'calibration', conv_file ), os.path.join( self.pkl_dir, 'calibration', cal_mjd_file )

        if os.path.isfile( conv_abs_path ):
            if self.verbose:
                print( "Loading previously saved conversion factor data..." )
            pickle_in = open( conv_abs_path, "rb" )
            conversion_factors = pickle.load( pickle_in )
            pickle_in.close()
            pickle_in = open( cal_abs_path, "rb" )
            cal_mjds = pickle.load( pickle_in )
            pickle_in.close()
        else:
            if self.verbose:
                print( "Making new conversion factor list..." )

            conversion_factors = []
            cal_mjds = []
            for e in self.get_closest_contfile():
                jpc = self.calculate_Jy_per_count( e )
                conversion_factors.append( jpc )
                cal_mjds.append( e[3] )

            conversion_factors = np.array( conversion_factors )

            if self.verbose:
                print( "Saving as {}".format( conv_file ) )

            pickle_out = open( conv_abs_path, "wb" )
            pickle.dump( conversion_factors, pickle_out )
            pickle_out.close()
            pickle_out = open( cal_abs_path, "wb" )
            pickle.dump( cal_mjds, pickle_out )
            pickle_out.close()


        if type( conversion_factors ) != np.ndarray:
            conversion_factors = np.array( conversion_factors )
        if type( cal_mjds ) != np.ndarray:
            cal_mjds = np.array( cal_mjds )

        print(conversion_factors)

        counter = 0

        for directory in self.dirs:
            for psr_file in sorted( os.listdir( directory ) ):
                try:
                    hdul, psr_mjd, psr_fe, obs_num, obs_mode = self.hdul_setup( directory, psr_file, False )
                    if self.verbose:
                        print( "Opening {}".format( psr_file ) )
                except OSError:
                    if self.verbose:
                        try:
                            print( "Couldn't open {}".format( psr_file ) )
                        except UnicodeEncodeError:
                            print( "Couldn't open {}".format( psr_file.encode( "utf-8" ) ) )
                    continue

                if obs_mode != "PSR":
                    continue

                ar = Archive( os.path.join( directory, psr_file ), verbose = self.verbose )
                data = ar.data_orig
                new_data = []
                for sub in data:
                    A, B, C, D = self.convert_subint_pol_state( sub, ar.subintheader[ 'POL_TYPE' ], "AABBCRCI", linear = ar.header[ 'FD_POLN' ] )
                    new_data.append( [ A, B ] )

                new_data = np.array( new_data )


                while psr_mjd != cal_mjds[ counter ]:
                    print( psr_mjd, cal_mjds[ counter ] )
                    counter += 1
                    if counter >= len( conversion_factors ):
                        break
                else:
                    for sub in new_data:
                        sub = conversion_factors[ counter ] * sub
                        print(sub.shape)
                    counter = 0

        return self

    #def get_cal_levels( self,  )


    # Save as ASCII text file
    def save( self, outroot = "cal_out", ext = '.ascii' ):
        outfile = outroot + ext
        with open( outfile, 'w' ) as f:
            for i, t in enumerate( self.omit ):
                for j, rej in enumerate( t ):
                    if rej == False:
                        f.write( str(i) + " " + str(self.ar.freq[i][j]) + "\n" )
        return outfile




if __name__ == "__main__":

    args = sys.argv[1:]
    c = Cal( args[0], args[1], args[2] )
    j = c.jpc( G = 11.0 )
