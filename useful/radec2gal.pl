#!/usr/bin/perl
# RA, DEC to l, b
# RA, DEC in HH:mm:ss

use strict;
use warnings;
use Math::Trig;

my ( $ra, $dec ) = @ARGV;

my ( $a_ngp, $d_ngp, $l_ngp ) = ( 192.85948 * ( pi/180 ), 27.12825 * ( pi/180 ), 122.93192 * ( pi/180 ) );

# Get everything into radians then convert

my ( $ra_hr, $ra_min, $ra_sec ) = split( ':', $ra );
my ( $dec_hr, $dec_min, $dec_sec ) = split( ':', $dec );
$ra = (15 * ( $ra_hr + ($ra_min / 60) + ($ra_sec / 3600) )) * ( pi/180 );
$dec = ( $dec_hr + ($dec_min / 60) + ($dec_sec / 3600) ) * ( pi/180 );

# Conversion

my $sinb = ( sin($d_ngp) * sin($dec) ) + ( cos($d_ngp) * cos($dec) * cos($ra - $a_ngp) );
my $b = asin($sinb);
my $sinl = ( cos($dec) * sin($ra - $a_ngp) ) / cos($b);
my $l = asin($sinl);
$l = ($l_ngp - $l) * ( 180/pi );
$b = $b * ( 180/pi );

print "$l $b\n"
