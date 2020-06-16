# ATNF Catalog data mining class
# Manchester, R. N., Hobbs, G. B., Teoh, A. and Hobbs, M., AJ, vol. 129, 1993-2006 (2005)

PLUS = '%2B'
MINUS = '-'
CNR = '+'

import requests
from fake_useragent import UserAgent
from bs4 import BeautifulSoup


class ATNF:

    def __init__( self, psrs ):
        self.psrs = psrs
        self.url = self.ATNF_srch_url()
        self.param_table = self.ATNF_get_table()
        self.names = self.param_list( 'PSRJ' )
        self.position = self.param_list( 'RAJ' )
        self.dm = self.param_list( 'DM' )

    def ATNF_srch_url( self ):
        psr_srch = ''
        for p in self.psrs:
            if '+' in p:
                q = p.replace( '+', PLUS )
            elif '-' in p:
                q = p.replace( '-', MINUS )
            else:
                q = p
            if p is not self.psrs[-1]:
                psr_srch += f'{q}{CNR}'
            else:
                psr_srch += f'{q}'

        url = f'https://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?version=1.63&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=&pulsar_names={psr_srch}&ephemeris=short&submit_ephemeris=Get+Ephemeris&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2=&style=Long+with+last+digit+error&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query'
        return url

    def ATNF_get_table( self ):
        ua = UserAgent()
        response = requests.get( self.url, {"User-Agent": ua.random} )
        soup = BeautifulSoup( response.text, "html.parser" )

        pre = soup.find_all( 'pre' )[0]
        pre = pre.text.split('\n')
        del pre[0]
        del pre[-1]
        return pre


    def param_list( self, param_key ):
        l = []
        for line in self.param_table:
            s = line.split()
            l.append( s )
        def get_DM_list():
            dm_ind = [ i for i, s in enumerate( self.param_table ) if 'DM' in s]
            disp = []
            for dm in dm_ind:
                if (l[dm][0] == 'DM1') or (l[dm][0] == 'DM2'):
                    continue
                disp.append( l[dm][1] )
            return disp
        def get_position():
            ra_ind, dec_ind = [ i for i, s in enumerate( self.param_table ) if 'RAJ' in s], [ i for i, s in enumerate( self.param_table ) if 'DECJ' in s]
            pos = []
            for ra, dec in zip( ra_ind, dec_ind ):
                pos.append( l[ra][1] + " " + l[dec][1] )
            return pos


        if param_key == 'DM':
            return get_DM_list()
        if param_key == 'RAJ' or param_key == 'DECJ':
            return get_position()

        ind = [ i for i, s in enumerate( self.param_table ) if param_key in s ]

        params = []
        for n in ind:
            params.append( l[n][1] )

        return params

    def make_pars( self, filename = None ):
        mode = self.param_list( 'BINARY' )

        i = 0
        for l in self.param_table:
            if l == '@-----------------------------------------------------------------':
                i += 1
            else:
                if filename == None:
                    parfile = f'{self.names[i]}_{mode[i]}.par'
                else:
                    parfile = filename
                with open( parfile, 'a+' ) as f:
                    f.write( f"{l}\n" )

        return parfile
