'''
Example of the gapless approximation code. Refers to the ring trap trap.
'''

import csv

try:
    from gapless import World
except:
    from trapsim.gapless import World

import numpy as np

'''
Add all the electrodes electrodes
'''

a = 1120e-6 #central electrode 'radius'
b = 2860e-6 #width of compensation electrodes


# y coords of the dc electrodes
y_ranges =  np.array([(a, 0),
            (a+b,a),
            (a+b, a),
            (a, 0),
            (0,-a),
            (-a, -a-b),
            (-a, -a-b),
            (0, -a),
            ])

# x coords of the dc electrodes
x_ranges =  np.array([(a, a+b),
            (0,a),
            (-a, 0),
            (-a-b, -a),
            (-a-b,-a),
            (-a, 0),
            (a, 0),
            (a+b, a),
            ])



it = 100
y_ranges1 = []
x_ranges1 = []
y_ranges2 = []
x_ranges2 = []
y_ranges3 = []
x_ranges3 = []
y_ranges4 = []
x_ranges4 = []
y_ranges5 = []
x_ranges5 = []
y_ranges6 = []
x_ranges6 = []
y_ranges7 = []
x_ranges7 = []
y_ranges8 = []
x_ranges8 = []

for i in range(1,it):
    x_ranges3.append((-a-(b*(i-1))/it,-a-(b*i)/it))
    y_ranges3.append((a+(b*i)/it,a+b))    
    
    x_ranges2.append((a+(b*(i-1))/it,a+(b*i)/it))
    y_ranges2.append((a+(b*i)/it,a+b))

    y_ranges4.append((a+(b*(i-1))/it,a+(b*i)/it))
    x_ranges4.append((-a-(b*i)/it,-a-b))
  
    y_ranges1.append((a+(b*(i-1))/it,a+(b*i)/it))
    x_ranges1.append((a+(b*i)/it,a+b))

    y_ranges5.append((-a-(b*(i-1))/it,-a-(b*i)/it))
    x_ranges5.append((-a-(b*i)/it,-a-b))

    y_ranges8.append((-a-(b*(i-1))/it,-a-(b*i)/it))
    x_ranges8.append((a+(b*i)/it,a+b))

    x_ranges6.append((-a-(b*(i-1))/it,-a-(b*i)/it))
    y_ranges6.append((-a-(b*i)/it,-a-b))

    x_ranges7.append((a+(b*(i-1))/it,a+(b*i)/it))
    y_ranges7.append((-a-(b*i)/it,-a-b))
                
xlist = [x_ranges1,x_ranges2,x_ranges3,x_ranges4,x_ranges5,x_ranges6,x_ranges7,x_ranges8]
ylist = [y_ranges1,y_ranges2,y_ranges3,y_ranges4,y_ranges5,y_ranges6,y_ranges7,y_ranges8]  
  
 
    
    
    
    

''' Now build your own world '''
w = World(1)
# first build the left electrodes
i=0
r = [0,0,185e-6]
for xr, yr in zip( x_ranges, y_ranges):
    i=i+1
    w.add_electrode(str(i), xr, yr, 'dc')
    w.dc_electrode_dict[str(i)].expand_in_multipoles(r)
    for xrp, yrp in zip(xlist[i-1], ylist[i-1]):
        w.dc_electrode_dict[str(i)].extend( [[ xrp,yrp ]] )    


C = w.multipole_control_matrix(r,['Ex','Ey','Ez','U3'], r0=1e-3)
       
f=open('ringtrapboxapproxMartix.csv','w')
for i in range(0,len(C[:,1])):
    np.savetxt(f, C[i,:], delimiter=",")
f.close()

w.drawTrap()


# add the center
#w.add_electrode('23', (cx1, cx2), (rfy1, rfy2), 'dc' )
# add the RF
#w.add_electrode('rf1', (smallrfx1, smallrfx2), (rfy1, rfy2), 'rf')
#w.add_electrode('rf2', (bigrfx1, bigrfx2), (rfy1, rfy2), 'rf')



