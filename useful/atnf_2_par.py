#!/usr/local/bin/python3
# ATNF to par converter
# Manchester, R. N., Hobbs, G. B., Teoh, A. and Hobbs, M., AJ, vol. 129, 1993-2006 (2005)

import sys
import atnf

if __name__ == "__main__":

    psrs = sys.argv[1:]
    atnf.make_pars( psrs )
