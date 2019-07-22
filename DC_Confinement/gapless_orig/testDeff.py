from gapless import *
import matplotlib.pyplot as plt
import numpy as np
w = World(axes_permutation=1)

""" run simplest working version of the sample gapless code from Haeffner lab github/trapsim
    The current version of the code gives coupling distances which are about a factor
    of 10 smaller than the values (presumably correct) listed in the example code."""



#from "effective distance - resonator". Note that the definition of the World class has changed
# since that code was written -- no longer initialized with RF frequency or distance scale (?)
w.add_electrode('test', (-5000e-6, 5000e-6), (-90e-6, 90e-6), 'dc', 1.0) # pickup electrode
xp = -100e-6;
yp = 0

plt.ion()

zwalk = np.arange(110e-6, 130e-6, 1e-6)
cnt = w.electrode_dict['test']
deff = []
for zp in zwalk:
    test = w.electrode_dict['test']
    deff.append(test.compute_d_effective([xp, yp, zp])[2]*1e6)    
plt.plot(zwalk*1e6, deff)

d0 = deff[0]

#their value of d0
dex0 = 352.414129741


plt.xlabel('vertical distance (um)')
plt.ylabel('Effective distance (um)')
plt.grid(True)
plt.show()


# from "effective distance"
w2 = World(axes_permutation=1)
w2.add_electrode('cnt', (-50e-6, 50e-6), (-50e-6, 50e-6), 'dc', 1.0) # pickup electrode
xp = 0.; yp = 0.;
zwalk = np.arange(.1e-6, 1e-6, 1e-7)
cnt = w2.electrode_dict['cnt']
deff = []
for zp in zwalk:
    
    deff.append(cnt.compute_d_effective((xp, yp, zp))[2]*1e3)

d1 = deff[0]

#their value of d1
dex1 = 55.536343101e-3

plt.figure()
plt.plot(zwalk*1e6, deff)
plt.xlabel('height, um')
plt.ylabel('coupling distance, mm')
plt.show()
    
    
#compare our computed coupling distances to theirs
print (dex0/d0), (dex0/d0) / (np.pi**2)
print (dex1/d1), (dex1/d1) / (np.pi**2)
