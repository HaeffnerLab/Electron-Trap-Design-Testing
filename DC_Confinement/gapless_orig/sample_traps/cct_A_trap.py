'''
Example of the gapless approximation code. Refers to the CCT A trap.
'''

try:
    from gapless import World
except:
    from trapsim.gapless import World

import numpy as np

'''
Add all the electrodes electrodes
'''
gap = 5.e-6

smallrfx1 = -185.e-6
smallrfx2 = -50.e-6
bigrfx1 = 50.e-6
bigrfx2 = 320.e-6
cx1 = -45.e-6
cx2 = 45.e-6

rfy1 = -5280.e-6
rfy2 = 9475.e-6

lx1 = -490.e-6
lx2 = -190.e-6
rx1 = 325.e-6
rx2 = 625.e-6

# y coords of the dc electrodes
y_ranges = 1e-6*np.array([(0, 300.),
            (305., 605.),
            (610., 910.),
            (915., 1215.),
            (1220., 1320.),
            (1325., 1625.),
            (1630., 1930.),
            (1935., 2235),
            (2240., 2540.),
            (2545., 2995.),
            (3000., 3450.)
            ])

''' Now build your own world '''
w = World()
# first build the left electrodes
for num, yr in zip( range(1, 12), y_ranges):
    w.add_electrode(str(num), (lx1, lx2), yr, 'dc')
# now the right electrodes
for num, yr in zip( range(12, 23), y_ranges):
    w.add_electrode(str(num), (rx1, rx2), yr, 'dc')
# add the center
w.add_electrode('23', (cx1, cx2), (rfy1, rfy2), 'dc' )
# add the RF
w.add_electrode('rf1', (smallrfx1, smallrfx2), (rfy1, rfy2), 'rf')
w.add_electrode('rf2', (bigrfx1, bigrfx2), (rfy1, rfy2), 'rf')
#w.drawTrap()
