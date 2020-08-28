#!/usr/local/bin/python3
# Plots pulsar time series

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from pypulse.archive import Archive
from physics import calculate_rms_matrix


if __name__ == "__main__":

    file = sys.argv[1]
    ar = Archive( file, prepare = True )
    ch = ar.getNsubint()
    chan = ar.getNchan()

    def on_key(event):
        print( event.key, math.floor(event.xdata), math.floor(event.ydata) )
        if event.key == 'z':
            with open( f'../Zap/{file[6:20]}_lbw_{file[-9:-5]}_2048.zap', 'a+' ) as t:
                t.write( f'{math.floor(event.xdata)} {ar.freq[math.floor(event.xdata)][math.floor(event.ydata)]}\n' )
        elif event.key == 'r':
            with open( f'../Zap/{file[6:20]}_lbw_{file[-9:-5]}_2048.zap', 'a+' ) as t:
                for n in range(ch):
                    t.write( f'{n} {ar.freq[n][math.floor(event.ydata)]}\n' )

    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, math.floor(event.xdata), math.floor(event.ydata)))
        with open( f'Zap/zap_{file}.ascii', 'a+' ) as t:
            t.write( f'{math.floor(event.xdata)} {ar.freq[math.floor(event.xdata)][math.floor(event.ydata)]}\n' )

    print("Subints / 2 = ", (ar.getNsubint()//2) + 1)

    data = ar.getData()

    mask = np.zeros( ar.getNbin() )
    np.set_printoptions( threshold = sys.maxsize )
    mask[ar.opw] = 1
    rms = np.array(calculate_rms_matrix( data, mask = mask ))
    rms_mean, rms_std = np.mean(rms), np.std( rms )
    data_lin = np.array([])
    sub_pol = ar.getNsubint() * ar.getNpol()
    num_profs = sub_pol * ar.getNchan()

    #for i in np.arange( ar.getNsubint() ):
    #    data_lin = np.append( data_lin, data[ i, 324, : ] )
    #data = np.reshape( data, ( num_profs * ar.getNbin() ) )


    # D_FAC = 32
    # for i in range(D_FAC):
    #     st, ed = i*(chan // D_FAC), (i + 1)*(chan // D_FAC)
    #     fig = plt.figure( figsize = (7, 7) )
    #     ax = fig.add_subplot(111)
    #     cmap = plt.cm.Blues
    #     ax.imshow( rms.T[st:ed, :], cmap = cmap, interpolation = 'nearest', extent = [ 0, ch, ed, st ], aspect = 'auto', norm = clr.Normalize( vmin = 0, vmax = np.amax(rms) ) )
    #     fig.colorbar( plt.cm.ScalarMappable( norm = clr.Normalize( vmin = 0, vmax = np.amax(rms) ), cmap = cmap ), ax = ax )
    #     cid = fig.canvas.mpl_connect('key_press_event', on_key)
    #
    #     #fig.canvas.mpl_disconnect(cid)
    #     #ax.plot( np.linspace( 1, num_profs, num = sub_pol * ar.getNbin() ), data_lin, linewidth = 0.1, color = 'k' )
    #     #ax.set_ylim( -630, 630 )
    #     #for i in np.arange( 1, num_profs + 1 ):
    #     #    if (i % ar.getNsubint()) == 0:
    #     #        ax.axvline( i, linewidth = 0.2, color = 'r' )
    #     #ax.set_xlim( 2400, 2600 )
    #
    #     plt.show()
    #     fig.canvas.mpl_disconnect(cid)

    ar.tscrunch()
    ar.fscrunch()
    ar.plot()
    exit()
    ar.imshow( origin = 'upper' )
