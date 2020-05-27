# ATNF mining tools
# Gets YMW16 distances from co-ordinates and DMs in the ATNF Catalog
# Can also use methods to mine ATNF data
# Manchester, R. N., Hobbs, G. B., Teoh, A. and Hobbs, M., AJ, vol. 129, 1993-2006 (2005)
# Must have YMW16 installed on local machine
# Yao, Manchester and Wang, Astrophysical Journal, vol. 835 (2017)

PLUS = '%2B'
MINUS = '-'
CNR = '+'

import sys
import os
import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup


def ATNF_psr_srch( psrs ):
    psr_srch = ''
    for p in psrs:
        if '+' in p:
            q = p.replace( '+', PLUS )
        elif '-' in p:
            q = p.replace( '-', MINUS )
        else:
            q = p
        if p is not psrs[-1]:
            psr_srch += f'{q}{CNR}'
        else:
            psr_srch += f'{q}'

    url = f'https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?version=1.63&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=&pulsar_names={psr_srch}&ephemeris=short&submit_ephemeris=Get+Ephemeris&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2=&style=Long+with+last+digit+error&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query'
    return url

def get_ATNF_table( url ):
    ua = UserAgent()
    response = requests.get( url, {"User-Agent": ua.random} )
    soup = BeautifulSoup( response.text, "html.parser" )

    pre = soup.find_all( 'pre' )[0]
    pre = pre.text.split('\n')
    del pre[0]
    del pre[-1]
    return pre


def make_pars( psrs ):
    url = ATNF_psr_srch( psrs )
    pre = get_ATNF_table( url )
    names, pos, dm = get_names_ra_dec_dm( psrs )
    mode = get_binary( psrs )

    i = 0
    for l in pre:
        if l == '@-----------------------------------------------------------------':
            i += 1
        else:
            parfile = f'{names[i]}_{mode[i]}.par'
            with open( parfile, 'a+' ) as f:
                f.write( f"{l}\n" )

    return pre

def get_binary( psrs ):
    url = ATNF_psr_srch( psrs )
    pre = get_ATNF_table( url )
    binary_ind = [ i for i, s in enumerate( pre ) if 'BINARY' in s]

    l = []
    for line in pre:
        s = line.split()
        l.append( s )

    modes = []
    for n in binary_ind:
        modes.append( l[n][1] )

    return modes

def get_names_ra_dec_dm( psrs ):
    url = ATNF_psr_srch( psrs )
    pre = get_ATNF_table( url )
    name_ind = [ i for i, s in enumerate( pre ) if 'PSRJ' in s]
    ra_ind, dec_ind = [ i for i, s in enumerate( pre ) if 'RAJ' in s], [ i for i, s in enumerate( pre ) if 'DECJ' in s]
    dm_ind = [ i for i, s in enumerate( pre ) if 'DM' in s]

    l = []
    for line in pre:
        s = line.split()
        l.append( s )

    names = []
    for n in name_ind:
        names.append( l[n][1] )
    pos = []
    for ra, dec in zip( ra_ind, dec_ind ):
        pos.append( l[ra][1] + " " + l[dec][1] )
    disp = []
    for dm in dm_ind:
        if (l[dm][0] == 'DM1') or (l[dm][0] == 'DM2'):
            continue
        disp.append( l[dm][1] )

    return names, pos, disp
