'''
Example of the gapless approximation code. Refers to the ring trap trap.
'''

import csv

try:
    from gapless import World
except:
    from trapsim.gapless import World

import numpy as np
import matplotlib.pyplot as plt

'''
Add all the electrodes electrodes
'''

a = 1120e-6 #central electrode 'radius'
b = 1860e-6 #width of compensation electrodes
r = a + b
it = 100 #number of iterations for boxes

h = r/((2**.5)*it) #outer step size

# y coords of the initial dc electrodes
y_ranges =  np.array([(0, h),
            (a,(r**2-h**2)**.5),
            (a,(r**2-h**2)**.5),
            (0, h),
            (0,-h),
            (-a,-(r**2-h**2)**.5),
            (-a,-(r**2-h**2)**.5),
            (0, -h),
            ])

# x coords of the initial dc electrodes
x_ranges =  np.array([(a, (r**2-h**2)**.5),
            (0,h),
            (0,-h),
            (-a, -(r**2-h**2)**.5),
            (-a, -(r**2-h**2)**.5),
            (0, -h),
            (0, h),
            (a, (r**2-h**2)**.5),
            ])


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

#coordinates of iterated electrodes
for i in range(1,it-1):
    y_ranges1.append((h*(i),h*(i+1)))
    x_ranges2.append((h*(i),h*(i+1)))
    x_ranges3.append((-h*(i),-h*(i+1)))
    y_ranges4.append((h*(i),h*(i+1)))
    y_ranges5.append((-h*(i),-h*(i+1)))
    x_ranges6.append((-h*(i),-h*(i+1)))
    x_ranges7.append((h*(i),h*(i+1)))
    y_ranges8.append((-h*(i),-h*(i+1)))
    if i < it*a/r:
        x_ranges1.append(((a**2-(h*(i))**2)**.5, (r**2-(h*(i+1))**2)**.5))
        y_ranges2.append(((a**2-(h*(i))**2)**.5, (r**2-(h*(i+1))**2)**.5))
        y_ranges3.append(((a**2-(h*(i))**2)**.5, (r**2-(h*(i+1))**2)**.5))  
        x_ranges4.append((-(a**2-(h*(i))**2)**.5, -(r**2-(h*(i+1))**2)**.5))
        x_ranges5.append((-(a**2-(h*(i))**2)**.5, -(r**2-(h*(i+1))**2)**.5))
        y_ranges6.append((-(a**2-(h*(i))**2)**.5, -(r**2-(h*(i+1))**2)**.5))
        y_ranges7.append((-(a**2-(h*(i))**2)**.5, -(r**2-(h*(i+1))**2)**.5))
        x_ranges8.append(((a**2-(h*(i))**2)**.5, (r**2-(h*(i+1))**2)**.5))   
    else:
        x_ranges1.append((h*(i+1), (r**2-(h*(i+1))**2)**.5))
        y_ranges2.append((h*(i+1), (r**2-(h*(i+1))**2)**.5))         
        y_ranges3.append((h*(i+1), (r**2-(h*(i+1))**2)**.5))  
        x_ranges4.append((-h*(i+1), -(r**2-(h*(i+1))**2)**.5))
        x_ranges5.append((-h*(i+1), -(r**2-(h*(i+1))**2)**.5))
        y_ranges6.append((-h*(i+1), -(r**2-(h*(i+1))**2)**.5))
        y_ranges7.append((-h*(i+1), -(r**2-(h*(i+1))**2)**.5)) 
        x_ranges8.append((h*(i+1), (r**2-(h*(i+1))**2)**.5))    


                
xlist = [x_ranges1,x_ranges2,x_ranges3,x_ranges4,x_ranges5,x_ranges6,x_ranges7,x_ranges8]
ylist = [y_ranges1,y_ranges2,y_ranges3,y_ranges4,y_ranges5,y_ranges6,y_ranges7,y_ranges8]  
  
 
    
    
    
    

''' Now build your own world '''
w = World(1)
# first build the dc electrodes
i=0
r = [0,0,185e-6]
for xr, yr in zip( x_ranges, y_ranges):
    i=i+1
    w.add_electrode(str(i), xr, yr, 'dc')
    for xrp, yrp in zip(xlist[i-1], ylist[i-1]):
        w.dc_electrode_dict[str(i)].extend( [[ xrp,yrp ]] )    
    w.dc_electrode_dict[str(i)].expand_in_multipoles(r)


C = w.multipole_control_matrix(r,['Ex','Ey','U3','U4'], r0=10**-3) #compute the control matrix
       
#write control matrix to file
f=open('ringtrapCMartix.csv','w')
for i in range(0,len(C[:,1])):
    np.savetxt(f, C[i,:], delimiter=",")
f.close()


#set electrode voltages to study potential frequencies from DC electrodes
for elname in range(1,9):
    
    w.set_voltage(str(elname),1e6*C[2,elname-1])
    print(w.electrode_dict[str(elname)].voltage)


#omega = []
#l=[]
#dist =2.1*10**-11
#n=1
#for i in range(0,n):
#    freq=w.compute_dc_potential_frequencies(np.add(r,[0,i*dist/n,i*dist/n]))
#    omega.append(freq[0])
#    l.append(i*dist/n)
#    print(i)
#    
#print ''

#print 'Trap Deformations ='
#print omega[0]
#print ''
#print 'Defmormation Directions ='
#print eigvec
#print ''

#omegax=[]
#omegay=[]
#omegaz=[]
#for el in omega:
#    omegax.append(el[0])
#    omegay.append(el[1])
#    omegaz.append(el[2])
#
#plt.plot(l,omegax,'ro',l,omegay,'bs',l,omegaz,'g^')
#plt.show(block = True)
#
#m = w.calculate_multipoles(['Ex','Ey','Ez','U1','U2','U3','U4','U5'])
#print 'Multipoles = '
#print m

#draw the trap
#w.drawTrap()




