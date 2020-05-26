#!/usr/local/bin/python3
# Pulsar and DNS classes to do DNS population studies
# Henryk T. Haniewicz 2020

# Imports
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import total_ordering

plt.rc( 'text', usetex = True )
plt.rc( 'font', family = 'serif' )
plt.rcParams[ "errorbar.capsize" ] = 2.5

from uncertainties import ufloat
from error_prop import calculate_errors_from_hist, hypotenuse_error

# CONSTANTS
R0_kpc = 8.122 # distance to galactic center in kpc
gal_rot_vel_kms = 233.3 # km/s
sun_pos = [ R0_kpc, 0.0, 0.02 ]
sun_vel = [ -11.1, 12.24 + gal_rot_vel_kms, 7.25 ]
c = 299792458.0
alpha_NGP = 192.85948 # deg
delta_NGP = 27.12825 # deg
l_NGP = 122.93192 # deg

# UNIT CONVERSIONS
kpc_to_m = 3.08567758149137e19
masyr_to_rads = 1.5362818500441604e-16
kpc_to_ltsec = kpc_to_m / c
R0 = R0_kpc * kpc_to_ltsec # Get R0 in lt-sec
gal_rot_vel_nodim = ( 1000 * gal_rot_vel_kms ) / c # Get galactic rotation velocity in dimensionless units
alpha_NGP_rad = alpha_NGP * ( math.pi / 180.0 )
delta_NGP_rad = delta_NGP * ( math.pi / 180.0 )
l_NGP_rad = l_NGP * ( math.pi / 180.0 )

@total_ordering
class Pulsar:

    def __init__( self, name, ra, dec, p0, p_dot, proper_motion, pm_direction, distance ):
        self.name = str( name )
        self.ra = ra # Degrees
        self.dec = dec # Degrees
        self.l, self.b = self.get_gal_coordinates() # Degrees
        self.p0 = p0 # Seconds
        self.p_dot = p_dot
        self.distance = distance # kpc
        # PMs in mas/yr
        if proper_motion is not None:
            if isinstance( proper_motion, float ) and pm_direction is None:
                raise ValueError( "If only the total proper motion is known, you must also provide the angle of its direction." )
            elif isinstance( proper_motion, float ) and pm_direction is not None:
                self.pm_direction = pm_direction * ( math.pi / 180.0 ) # Input is degrees but we need it in radians
                self.pmtot = proper_motion
                self.pmra = -self.pmtot * math.sin( self.pm_direction )
                self.pmdec = self.pmtot * math.cos( self.pm_direction )
            elif len( proper_motion ) == 2: # assuming in RA and DEC
                self.pm_direction = None
                self.pmra = proper_motion[0] * math.cos( self.dec * ( math.pi / 180.0 ) )
                self.pmdec = proper_motion[1]
                self.pmtot = math.sqrt( self.pmra**2 + self.pmdec**2 )
            else:
                raise ValueError( "Make changes to your input." )

            self.xyz_position = self.get_xyz_pos()
            self.xyz_velocity = self.get_xyz_vel()
            self.predicted_vertical_velocity = self.xyz_velocity[2] - sun_vel[2]
            self.predicted_vt = self.predict_tangential_vel()
            self.predicted_pms = self.predict_proper_motions()
            self.pm_pec = self.get_peculiar_proper_motion()
            self.peculiar_tangential_velocity, self.vt_err = self.estimate_tangential_vel_gaussian()
            self.peculiar_radial_velocity, self.vr_err = self.estimate_radial_vel()
            self.total_peculiar_velocity, self.vtot_err = self.get_total_peculiar_velocity()
        else:
            self.pmtot = proper_motion

    def __eq__( self, other ):
        return ( self.name[1:].lower() == other.name[1:].lower() )

    def __lt__( self, other ):
        return ( self.name[1:].lower() < other.name[1:].lower() )


    def get_gal_coordinates( self ):
        sinb = ( math.sin( delta_NGP_rad ) * math.sin( self.dec * ( math.pi / 180.0 ) ) ) + ( math.cos( delta_NGP_rad ) * math.cos( self.dec * ( math.pi / 180.0 ) ) * math.cos( ( self.ra * ( math.pi / 180.0 ) ) - alpha_NGP_rad ) )
        b = math.asin( sinb ) # Radians
        sin_lngpl = ( math.cos( self.dec * ( math.pi / 180.0 ) ) * math.sin( ( self.ra * ( math.pi / 180.0 ) ) - alpha_NGP_rad ) ) / math.cos( b )
        lngpl = math.asin( sin_lngpl )
        l = l_NGP_rad - lngpl # Radians
        b = b * ( 180.0 / math.pi )
        l = l * ( 180.0 / math.pi )

        return l, b # Degrees

    def get_xyz_pos( self ):
        # Gets the X, Y and Z co-ordinates as well as PSR's distance to Galactic center
        x = R0_kpc - ( self.distance * math.cos( self.l * ( math.pi / 180.0 ) ) * math.cos( self.b * ( math.pi / 180.0 ) ) )
        y = self.distance * math.cos( self.l * ( math.pi / 180.0 ) ) * math.cos( self.b * ( math.pi / 180.0 ) )
        z = sun_pos[2] + ( self.distance * math.sin( self.b * ( math.pi / 180.0 ) ) )
        r = math.sqrt( x**2 + y**2 + z**2 )

        return x, y, z, r # Galactocentric?

    def get_xyz_vel( self ):
        # Gets the X, Y and Z velocity components
        vx = -( gal_rot_vel_kms * self.xyz_position[1] ) / self.xyz_position[3]
        vy = ( gal_rot_vel_kms * self.xyz_position[0] ) / self.xyz_position[3]
        vz = 0.0

        return vx, vy, vz

    def predict_tangential_vel( self ):
        # Predicts the tangential velocity contribution due to the LSR
        return ( -( ( self.xyz_velocity[0] - sun_vel[0] ) * ( self.xyz_position[1] - sun_pos[1] ) ) / self.distance ) + ( ( ( self.xyz_velocity[1] - sun_vel[1] ) * ( self.xyz_position[0] - sun_pos[0] ) ) / self.distance )

    def predict_proper_motions( self ):
        # Predicts the proper motion contribution due to the LSR
        mu_horizontal = self.predicted_vt / ( 4.740470463533348 * self.distance )
        mu_vertical = self.predicted_vertical_velocity / ( 4.740470463533348 * self.distance )

        return mu_horizontal, mu_vertical

    def get_peculiar_proper_motion( self ):
        # Gets the peculiar proper motion (w.r.t LSR)
        mu_horizontal_pec = self.pmra - self.predicted_pms[0]
        mu_vertical_pec = self.pmdec - self.predicted_pms[1]
        mu_pec = math.sqrt( mu_horizontal_pec**2 + mu_vertical_pec**2 )

        return mu_pec

    def estimate_tangential_vel_gaussian( self, iterations = 20000, show_hist = False ):
        # Picks v_trans from a Gaussian with 1-sigma (68% confidence) width = 20% of the distance, Yao et al. 2016 then multiplies with peculiar motion
        distribution = np.array([])
        for _ in range( iterations ):
            vt = 4.740470463533348 * np.random.normal( self.distance, 0.2*self.distance ) * self.pm_pec
            distribution = np.append( distribution, vt )
        median = np.median( distribution )
        d, n, _ = plt.hist( distribution, bins = math.floor(iterations/20), density = True, color = 'r' )
        err1, err2 = calculate_errors_from_hist( d, n, 68, show = True, color = 'k', linestyle = '--', linewidth = 1 )
        if show_hist:
            plt.xlabel( r"Tangential velocity (km s$^{-1}$)", fontsize = 20 )
            plt.yticks([])
            plt.show()

        plt.close()

        return median, [ err1, err2 ]

    def estimate_radial_vel( self, iterations = 50000, show_hist = False ):
        # Picks an angle uniform is cos(theta) then multiplies median transverse velocity by cot(theta)
        distribution = np.array([])
        for _ in range( iterations ):
            theta = math.acos( np.random.uniform( 0, 1 ) )
            vr = self.peculiar_tangential_velocity * ( 1 / math.tan( theta ) )
            distribution = np.append( distribution, vr )
        median = np.median( distribution )
        d, n, _ = plt.hist( distribution, bins = math.floor(iterations/20), density = True, color = 'r' )
        err1, err2 = calculate_errors_from_hist( d, n, 68, show = True, color = 'k', linestyle = '--', linewidth = 1 )
        if show_hist:
            plt.xlim(0, 250)
            plt.xlabel( r"Radial velocity (km s$^{-1}$)", fontsize = 20 )
            plt.yticks([])
            plt.show()

        plt.close()

        return median, [ err1, err2 ]

    def get_total_peculiar_velocity( self ):
        # Assumes independence of variables which is clearly false but will do.
        vtot = math.sqrt( self.peculiar_tangential_velocity**2 + self.peculiar_radial_velocity**2 )
        vh = hypotenuse_error( [ self.peculiar_tangential_velocity, self.peculiar_radial_velocity ], [ self.vt_err[1], self.vr_err[1] ] )
        vl = hypotenuse_error( [ self.peculiar_tangential_velocity, self.peculiar_radial_velocity ], [ self.vt_err[0], self.vr_err[0] ] )

        err = [vl, vh]

        return vtot, err

    def display_header_information( self ):
        print( f"Distance to Earth (kpc): \t\t\t{self.distance}\t+/- {0.2*self.distance:.3f}" )
        print( f"Position in (\u03B1, \u03B4) (\u00B0): \t\t\t({self.ra}, {self.dec})" )
        print( f"Position in (l, b) (\u00B0): \t\t\t({self.l:.3f}, {self.b:.3f})" )
        print( f"Spin period (s): \t\t\t\t{self.p0}" )
        print( f"Spin period derivative: \t\t\t{self.p_dot}" )
        print( f"Observed total proper motion (mas/yr): \t\t{self.pmtot:.3f}\n" )


    def display_velocity_information( self, header = False ):
        if header:
            self.display_header_information()
        if self.pmtot is None:
            print( "No velocity information" )
            return

        print( f"Distance to the Galactic center (kpc): \t\t{self.xyz_position[3]:.3f}" )
        print( f"Predicted transverse velocity (km/s): \t\t{self.predicted_vt:.3f}" )
        print( f"LSR total proper motion (mas/yr): \t\t{self.pm_pec:.3f}" )
        print( f"LSR tangential velocity (km/s): \t\t{self.peculiar_tangential_velocity:.3f}\t( -{self.vt_err[0]:.3f}, +{self.vt_err[1]:.3f} )" )
        print( f"LSR radial velocity (km/s): \t\t\t{self.peculiar_radial_velocity:.3f}\t( -{self.vr_err[0]:.3f}, +{self.vr_err[1]:.3f} )" )
        print( f"Total peculiar velocity (km/s): \t\t{self.total_peculiar_velocity:.3f}\t( -{self.vtot_err[0]:.3f}, +{self.vtot_err[1]:.3f} )" )
        print( "----------------------------------------------------------------------------------\n" )


class DNS( Pulsar ):

    def __init__( self, name, ra, dec, p0, p_dot, proper_motion, pm_direction, distance, a1, pb, eccentricity, m2, mtot ):
        super().__init__( name, ra, dec, p0, p_dot, proper_motion, pm_direction, distance )
        self.a1 = a1 # lt-sec
        self.pb = pb # Days
        self.eccentricity, self.ecc_err = eccentricity.n, eccentricity.s
        self.m2, self.m2_err = m2.n, m2.s # Solar masses
        self.mtot, self.mtot_err = mtot.n, mtot.s # Solar masses
        self.m1, self.m1_err = (mtot - m2).n, (mtot - m2).s


    def display_header_information( self ):
        super().display_header_information()
        print( f"Binary parameters:" )
        print( f"Projected semi-major axis (lt-sec): \t\t{self.a1}" )
        print( f"Orbital period (days): \t\t\t\t{self.pb}" )
        print( f"Eccentricity: \t\t\t\t\t{self.eccentricity}\t+/- {self.ecc_err}" )
        print( f"Pulsar Mass (M\u2299): \t\t\t\t{self.m1:.5f}\t+/- {self.m1_err:.5f}" )
        print( f"Companion Mass (M\u2299): \t\t\t\t{self.m2}\t+/- {self.m2_err}\n" )

def create_m2_vel_array( psrs ):
    companion_masses, m_err, velocities, v_err_l, v_err_h = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for psr in psrs:
        if isinstance( psr, DNS ):
            companion_masses = np.append( companion_masses, psr.m2 )
            m_err = np.append( m_err, psr.m2_err )
            velocities = np.append( velocities, psr.total_peculiar_velocity )
            v_err_l, v_err_h = np.append( v_err_l, psr.vtot_err[0] ), np.append( v_err_h, psr.vtot_err[1] )

    v_err = np.array([ v_err_l, v_err_h ])

    return companion_masses, m_err, velocities, v_err

def create_m2_ecc_array( psrs ):
    companion_masses, m_err, eccentricities = np.array([]), np.array([]), np.array([])
    for psr in psrs:
        if isinstance( psr, DNS ):
            companion_masses = np.append( companion_masses, psr.m2 )
            m_err = np.append( m_err, psr.m2_err )
            eccentricities = np.append( eccentricities, psr.eccentricity )

    return companion_masses, m_err, eccentricities
