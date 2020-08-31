#!/usr/local/bin/python3
# Bins down training and validation files

import sys, os, subprocess
import numpy as np
import random
from nn import read_data
import time

if len(sys.argv) == 1:
    print('''Bins down training and validation files

Use: bindown_nntraining_sets.py [training file] [validation file]

where double [[]] are optional arguments.
    ''')
    exit(1)

BDO_tr = 150
BDO_val = int( BDO_tr * 0.2 )
#BDR = 0.5

training, validation = sys.argv[1], sys.argv[2]

tr_o = training.split('.')[0] + "_bd." + training.split('.')[1]
val_o = validation.split('.')[0] + "_bd." + validation.split('.')[1]

t0 = time.time()

x, y, xv, yv = read_data( training, validation )

p0, p1 = [], []



for i, record in enumerate( y.T ):
    print( f'{100*(i/len(y.T)):.3f}%', end = '\r')
    p = np.concatenate( (x.T[i], record ) )
    if record == 0:
        p0.append( p )
    elif record == 1:
        p1.append( p )
    else:
        raise ValueError( "Must be either 0 or 1" )

# Pick as many as needed from each partition
subgroup0 = random.sample( p0, k = BDO_tr )
subgroup1 = random.sample( p1, k = BDO_tr )

# Combine and shuffle the results
combined = subgroup0 + subgroup1
random.shuffle(combined)

with open( tr_o, 'w+' ) as t:
    t.write( f'# Binned down training set for {training}\n' )
    for prof in combined:
        np.savetxt( t, prof, fmt = '%1.5f ', newline = '' )
        t.write("\n")


''' Validation code starts here! '''

p0, p1 = [], []

for i, record in enumerate( yv.T ):
    print( f'{100*(i/len(yv.T)):.3f}%', end = '\r')
    p = np.concatenate( (xv.T[i], record ) )
    if record == 0:
        p0.append( p )
    elif record == 1:
        p1.append( p )
    else:
        continue

# Pick as many as needed from each partition
subgroup0 = random.sample( p0, k = BDO_val )
subgroup1 = random.sample( p1, k = BDO_val )

# Combine and shuffle the results
combined = subgroup0 + subgroup1
random.shuffle(combined)

with open( val_o, 'w+' ) as t:
    t.write( f'# Binned down validation set for {validation}\n' )
    for prof in combined:
        np.savetxt( t, prof, fmt = '%1.5f ', newline = '' )
        t.write("\n")

t1 = time.time()
te = t1 - t0
print( f'Time taken = {te:.2f}s' )
