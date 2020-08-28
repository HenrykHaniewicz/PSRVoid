#!/usr/local/bin/python3
# Combines two training / validation sets from root names
# New ASCII header relates to the two file roots chosen

import sys

def combine_sets( file1, file2, out = "combined_set.out", type = 't' ):
    with open( out, 'w' ) as f:
        if type == 't':
            f.write( f"# NN training data created from {file1} and {file2}\n" )
        elif type == 'v':
            f.write( f"# NN validation data created from {file1} and {file2}\n" )
        else:
            f.write( f"# Combined NN data from {file1} and {file2}\n" )

        with open( file1, 'r' ) as f1:
            next(f1)
            for line in f1:
                f.write( line )
        with open( file2, 'r' ) as f2:
            next(f2)
            for line in f2:
                f.write( line )
    return 0

if __name__ == "__main__":

    o_t = "J1829+2456_58402+58404+58406+58408_430_2048.training"
    o_v = "J1829+2456_58402+58404+58406+58408_430_2048.validation"

    t1 = sys.argv[1] + ".training"
    v1 = sys.argv[1] + ".validation"
    t2 = sys.argv[2] + ".training"
    v2 = sys.argv[2] + ".validation"

    combine_sets( t1, t2, o_t )
    combine_sets( v1, v2, o_v, 'v' )
