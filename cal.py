#!/usr/local/bin/python3
# New calibrator

"""
Saves as ASCII, all values in Jy/count:         Freq (MHz)         AA          BB         CR         CI
"""

import sys
import os
import numpy as np
import flux

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as unit

format = 2

class Cal():

    def __init__( self, file, cont_name, cont_fits_dir = None, verbose = False, **kwargs ):
        self.file = file
        self.cont_name = cont_name
        self.verbose = verbose
        self.ar = Archive( file, prepare = False, verbose = False )
        self.ar.dedisperse( wcfreq = self.ar.wcfreq )
        self.ar.center()


    def get_onoff_list( self, tolerance = 1 ):

        try:
            on, off = np.loadtxt( "logs/on_off.log" )
        except Exception:
            on, off = [], []

            # Get accurate co-ordinates for source
            pos, params = flux.find_source_params_f2( self.cont_name )
            m_coordinates = SkyCoord( f"{pos[0]} {pos[1]}", unit = ( unit.hourangle, unit.degree ) )

            for f in sorted( os.listdir( self.cont_dir ) ):
                continuum_file = os.path.join( self.cont_dir, f )

                try:
                    hdul, mjd, fe, cont_ra, cont_dec, obs_mode = u.find_fe_mjd_ra_dec( continuum_file )
                except Exception:
                    continue

                # Get co-ordinates in FITS file
                coords = SkyCoord( f"{cont_ra} {cont_dec}", unit = ( unit.hourangle, unit.degree ) )

                onoff_list.append( { 'MJD' : mjd, 'ON' : None, 'OFF' : None, 'FE' : fe, 'NUM' : obs_num } )

                # Determine if co-ordinates signify ON or OFF source observation
                if m_coordinates.separation( coords ) <= ( tolerance * unit.arcmin ):
                    mode = 'ON'
                else:
                    mode = 'OFF'

                for dict in onoff_list:
                    if (dict[ 'MJD' ] == mjd) and (dict[ 'FE' ] == fe) and (dict[ 'NUM' ] == obs_num) and (dict[ mode ] is None):
                        dict[ mode ] = file

            for dict in reversed( onoff_list ):
                if (dict[ 'ON' ] is None) or (dict[ 'OFF' ] is None):
                    onoff_list.remove( dict )

            if self.verbose:
                print( "Saving as {}".format( dict_file ) )

            pickle_out = open( abs_dict_file, "wb" )
            pickle.dump( onoff_list, pickle_out )
            pickle_out.close()

        # Returns
        return on, off

    def get_closest_contfile( self, mjd_tol = 50 ):

        onoff_list = self.get_onoff_list( tolerance = 1 )

        a = []

        for directory in self.dirs:
            for psr_file in sorted( os.listdir( directory ) ):

                mjd_list = []

                try:
                    hdul, psr_mjd, psr_fe, obs_num, obs_mode = self.hdul_setup( directory, psr_file )
                    if self.verbose:
                        print( "Opening {}".format( psr_file ) )
                except OSError:
                    if self.verbose:
                        print( "Couldn't open {}".format( psr_file ) )
                    continue

                psr_file = os.path.join( directory, psr_file )

                if obs_mode == 'CAL':

                    for dict in onoff_list:
                        if dict[ 'FE' ] == psr_fe:
                            delta_mjd = abs( dict[ 'MJD' ] - psr_mjd )
                            if all( elem > delta_mjd for elem in mjd_list ):
                                mjd_list.append( delta_mjd )
                                if delta_mjd < mjd_tol:
                                    on = os.path.join( self.cont_dir, dict[ 'ON' ] )
                                    off = os.path.join( self.cont_dir, dict[ 'OFF' ] )
                                else:
                                    on, off = [None, None]
                    a.append( [psr_file, on, off, psr_mjd] )

        return a


    def calculate_Jy_per_count( self, cal_file_list ):

        """
        Input list: [ PSR_CAL, ON_CAL, OFF_CAL, CAL_MJD ]

        Returns:
        conversion_factor  :  np.ndarray
        """

        G = 11.0

        if type( cal_file_list ) != np.ndarray:
            cal_file_list = np.array( cal_file_list )

        if cal_file_list.ndim != 1:
            raise ValueError( "Should be a vector" )

        archives = []
        freqs = []
        for i, f in enumerate( cal_file_list[:-1] ):
            hdul = fits.open( f )
            freqs.append( hdul[3].data[ 'DAT_FREQ' ][0] )
            hdul.close()
            archives.append( Archive( f, prepare = False, verbose = self.verbose ) )


        aabb_list = []

        for i, arc in enumerate( archives ):
            arc.dedisperse( wcfreq = archives[i].wcfreq )
            arc.center()

            arc.tscrunch()
            A, B, C, D = self.convert_subint_pol_state( arc.getData(), arc.subintheader[ 'POL_TYPE' ], "AABBCRCI", linear = arc.header[ 'FD_POLN' ] )
            l = { 'ARC' : arc, 'DATA' : [ A, B ], 'FREQS' : freqs[i], 'S_DUTY' : arc.getValue( 'CAL_PHS' ) , 'DUTY' : arc.getValue( 'CAL_DCYC' ), 'BW' : arc.getBandwidth() }
            aabb_list.append( l )


        H, L, T0 = self._prepare_calibration( aabb_list )
        F_ON = ( H[1]/L[1] ) - 1
        F_OFF = ( H[2]/L[2] ) - 1

        C0 = T0[1:] / ( ( 1 / F_ON ) - ( 1 / F_OFF ) )
        T_sys = C0 / F_OFF
        F_cal = ( T_sys * F_OFF ) / G


        Fa, Fb = interp1d( aabb_list[1][ 'FREQS' ], F_cal[0][0], kind='cubic', fill_value = 'extrapolate' ), interp1d( aabb_list[2][ 'FREQS' ], F_cal[0][1], kind='cubic', fill_value = 'extrapolate' )

        conversion_factor = [ np.array(Fa( aabb_list[0][ 'FREQS' ] ) / ( H[0][0] - L[0][0] )), np.array( Fb( aabb_list[0][ 'FREQS' ] ) / ( H[0][1] - L[0][1] ) ) ]
        conversion_factor = np.array( conversion_factor )

        return conversion_factor


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


    def convert_subint_pol_state( self, subint, input, output, linear = 'LIN' ):

        """

        """

        if input == output:
            out_S = subint
        elif input == "AABBCRCI" and output == "IQUV": # Coherence -> Stokes
            A, B, C, D = subint
            if linear == 'LIN':
                I = A+B
                Q = A-B
                U = 2*C
                V = 2*D
            else:
                I = A+B
                Q = 2*C
                U = 2*D
                V = A-B
            out_S = [ I, Q, U, V ]
        elif input == "IQUV" and output == "AABBCRCI": # Stokes -> Coherence
            I, Q, U, V = subint
            if linear == 'LIN':
                A = (I+Q)/2.0
                B = (I-Q)/2.0
                C = U/2.0
                D = V/2.0
            else:
                A = (I+V)/2.0
                B = (I-V)/2.0
                C = Q/2.0
                D = U/2.0
            out_S = [ A, B, C, D ]
        else:
            print("WTF")


        out_S = np.array( out_S )

        return out_S


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



    print("Under construction")
