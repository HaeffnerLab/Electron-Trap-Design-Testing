
"""A notebook to define the x/y geometry of the F trap for the resonator 
    experiment. 
       y = dimension along the center electrode longest axis
       x = the orthogonal direction to y in the plane of the trap
       z = normal to x/y, out of the trap plane
       
       dimensions are from the F_nowire_correct.dxf autocad file in the 
       resonator-cooling/trap_simulations shared directory. The origin
	   is located on the trap axis, with y=0 at the bottom edges 
	   of the two lowermost electrodes (1 and 11)
       
       all dimensions are in meters
       
       2015-09-21"""


#check that the gapless module is present
try:
    from gapless import World
except ImportError:
    print "gapless module not present"
    exit(-1)
    
import numpy as np
import matplotlib.pyplot as plt

#define the electrode dimensions

#1um in mks
u = 1E-6

#width of non-center electrode in the x-direction
dx = 200*u

#height of non-center electrode in the y-direction
dy  = 200*u

#gap between adjacent electrode edges in the y direction
gapy = 10*u

#center electrode x-range (1 = lower value, 2 = upper value)
centerxRange = (-80*u, 80*u)

# center electrode y-range
centeryRange = (-2955*u, 5395*u)

#for the RF electrodes: the two long strips bordering the center
RF1xRange = (-150*u, -90*u)
RF2xRange = (90*u, 150*u)

RFyRange = (-2965*u, 5295*u)


y_indices = np.linspace(0, 9, 10)

lower_y_edges = y_indices * (gapy + dy)


upper_y_edges = lower_y_edges + dy


#y-range tuples for the outer electrodes
yranges = list(map(lambda i: (lower_y_edges[i], upper_y_edges[i]), y_indices))

xrangeRight = (160*u,160*u + dx)
xrangeLeft = (-160*u - dx, -160*u)



#create a world object and start filling in electrodes
#axes_permutation = 1 means that y lies in the trap plane, and z is the normal
# direction (height)
FTrapWorld = World(axes_permutation=1)

#left electrodes, 1-10
for i in range(1, 11):
    yrange = yranges[i-1]
    FTrapWorld.add_electrode(str(i), xrangeLeft, yrange, 'dc')
    FTrapWorld.add_electrode(str(i+10), xrangeRight, yrange, 'dc')	
	
FTrapWorld.add_electrode('21', centerxRange, centeryRange, 'dc')
FTrapWorld.add_electrode('RF1', RF1xRange, RFyRange, 'rf')
FTrapWorld.add_electrode('RF2', RF2xRange, RFyRange, 'rf')

plt.ion()
FTrapWorld.drawTrap()


