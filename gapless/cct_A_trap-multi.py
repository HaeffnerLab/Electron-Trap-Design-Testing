# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

u = loadtxt('../u_6_20.txt')
import seaborn as sns
        

# <codecell>

run '../ring_trap'

# <codecell>

zp = 760e-6
yp = 107.e-6
xwalk = np.arange(-30e-6, 0e-6, 1e-6)
pot = []
w.set_omega_rf(2*pi*35e6)
w.set_voltage('rf1', 50.)
w.set_voltage('rf2', 50.)
for xp in xwalk:
    r = (xp, yp, zp)
    pot.append( w.compute_pseudopot([xp, yp, zp]) )
plot(array(xwalk)*1e6, array(pot))

# <codecell>

from scipy import optimize as opt
def find_trap_location(r0):
    x0 = [r0[0], r0[1]]
    fixed = r0[2]
    cost = lambda r: w.compute_pseudopot([r[0],r[1],fixed])
    result = opt.minimize(cost, x0, method='Nelder-Mead')
    r_trap = [result.x[0], result.x[1], fixed ]
    return array(r_trap)

r0 = array([-16, 102, 760])*1e-6
r_trap = find_trap_location(r0)
print r_trap*1e6
print w.compute_pseudopot(r_trap)

# <codecell>

Nz =100
#Z_center = 760e-6
Z_center = 1270e-6
Z = linspace(Z_center-200e-6,Z_center+200e-6,Nz)

elec_name = w.dc_electrode_dict.keys()
elec_name.sort()
mult_index = 1

for en in elec_name:
    i = int(en)
    #print i, u[i,mult_index]
    w.set_voltage(en, u[i,mult_index])
   
potlist2 = []
for z_i in xrange(Nz):
    r = [xp,yp,Z[z_i]]
    a = w.compute_total_dc_potential(r)
    potlist2.append(a)
potlist2 = array(potlist2)

# <codecell>

elec_name = w.dc_electrode_dict.keys()
elec_name.sort()
mult_index = 1

zp = Z_center
yp = 107e-6
xp = -17e-6

r = [xp,yp,zp]
for en in elec_name:
    i = int(en)
    w.set_voltage(en, 1)
    #a = w.compute_total_dc_potential
    aa = w.dc_electrode_dict[en]
    #v = aa.compute_voltage(r)
    w.set_voltage(en, 0)
   
potlist2 = []
for z_i in xrange(Nz):
    r = [xp,yp,Z[z_i]]
    a = w.compute_total_dc_potential(r)
    potlist2.append(a)
potlist2 = array(potlist2)

# <codecell>

vdict = {}
vdict1 = {}
mult_arr = zeros((len(elec_name),5))
for en in elec_name:
    w.set_voltage(en, 1.)
    i = int(en) - 1
    zp = Z_center
    yp = 107e-6
    w.dc_electrode_dict[en].expand_in_multipoles([xp, yp, zp ], r0=1)
    x = w.dc_electrode_dict[en]
    z4 = x.multipole_dict['z^4']
    u2 = x.multipole_dict['U2']
    Ex = x.multipole_dict['Ex']
    Ey = x.multipole_dict['Ey']
    Ez = x.multipole_dict['Ez']
    vdict[en] = z4
    vdict1[en] = u2
    mult_arr[i,:] = [Ex/1e3,Ey/1e3,Ez/1e3,u2/1e6,z4/1e15]
    w.set_voltage(en, 0.)
  

# <codecell>

shorted = [0,11]
mult_arr[shorted,:] = 0
useM = 5
C = []
for mult in range(useM):
    B = np.zeros(useM)
    B[mult] = 1
    A = np.linalg.lstsq(mult_arr[:,0:useM].transpose(),B)[0]
    C.append(A)
C = array(C)

# <codecell>

mult_num = 4
figure()
plot(C[mult_num,0:11])
figure()
plot(C[mult_num, 11:22])
#print C[0,0:10]
#print C[0,10]

# <codecell>

mult_num = 3
for en in elec_name:
    i = int(en)
    v = C[mult_num, i - 1]
    w.set_voltage(en, v)
Y_center = 107e-6
X_center = -17e-6

Nstep = 100
zwalk = linspace(Z_center-400e-6,Z_center+400e-6,Nstep)
ywalk = linspace(Y_center-20e-6,Y_center+20e-6,Nstep)
xwalk = linspace(X_center-20e-6,X_center+20e-6,Nstep)
pot = []
for xp in xwalk:
    r = [xp, Y_center, Z_center]
    pot.append( w.compute_total_dc_potential(r) )
figure()
plot(np.array(xwalk)*1e6, pot)
title('xwalk')
pot = []
for yp in ywalk:
    r = [X_center, yp, Z_center]
    pot.append( w.compute_total_dc_potential(r) )
figure()
plot(np.array(ywalk)*1e6, pot)
title('ywalk')

pot = []
for zp in zwalk:
    r = [X_center, Y_center, zp]
    pot.append( w.compute_total_dc_potential(r) )
figure()
#plot(np.log(np.array(zwalk)*1e6), abs(np.log(np.array(pot))))
plot((zwalk - Z_center)*1e6, pot)
title('zwalk')
result =  np.polyfit((zwalk-Z_center), pot,4)
r0 = 50e-6
rn  = [r0**n for n in range(5)]
rn.reverse()
rn = np.array(rn)
energy_vec =  result*rn
print energy_vec
figure()
ratio_vec = []
for zp in zwalk:
    r0 = zp - Z_center
    rn  = [r0**n for n in range(5)]
    rn.reverse()
    rn = np.array(rn)
    energy_vec =  result*rn
    #print energy_vec
    e4 = energy_vec[0]
    e2 = energy_vec[2]
    ratio_vec.append(e4/e2)
plot((zwalk - Z_center)*1e6, ratio_vec)

# <codecell>

matshow(abs(C[[3],:]))
#matshow(abs(C[[4],:]))
colorbar()

# <codecell>

sup = 10

for en in elec_name:
    i = int(en) - 1
    w.set_voltage(en, C[3,i])
potlist3 = []
for z_i in xrange(Nz):
    r = [xp,yp,Z[z_i]]
    a = w.compute_total_dc_potential(r)
    potlist3.append(a)
potlist3 = array(potlist3) #/ max(np.abs(potlist3)) 
potlist3 = (potlist3) - min((potlist3))


for en in elec_name:
    i = int(en) - 1
    w.set_voltage(en, C[3,i])
potlist4 = []
for z_i in xrange(Nz):
    r = [xp,yp,Z[z_i]]
    a = w.compute_total_dc_potential(r)
    potlist4.append(a)
potlist4 = array(potlist4) #/ np.max(np.abs(potlist4))
potlist4 = (potlist4) - np.min((potlist4))

for en in elec_name:
    i = int(en) - 1
    w.set_voltage(en, C[3,i]-3.1*C[3,i])  
potlist5 = []
for z_i in xrange(Nz):
    r = [xp,yp,Z[z_i]]
    a = w.compute_total_dc_potential(r)
    potlist5.append(a)       
potlist5 = array(potlist5) #/ np.max(np.abs(potlist4))
potlist5 = (potlist5) - np.min((potlist5))

r0 =1.0
pot4 = -1*(r0**4)*(35 * (Z-Z_center)**4 - 30 * (Z-Z_center)**2 / r0**2 + 3 / r0**4)

clf()
plot(Z,potlist3,'b',label='2nd order')
#plot(Z,potlist4,'r',label='4th order')
#plot(Z,pot4,'k',label='4th order ideal')
#plot(Z,6e4*(Z-Z_center)**2,'k',label='2nd order ideal')
#plot(Z,potlist5/2,'c',label='Superpos: '+str(sup))
legend()

# <codecell>

cfile_index = [0,1,2,4,5]
cfile = zeros((8,23))
index1 = 0
for i in cfile_index:
    cfile[i,:] = C[index1,:]
    index1 += 1
cout = reshape(cfile,(23*8,1))
np.savetxt("Atrap_el5_fourthorder.txt", cout)

# <codecell>

import numdifftools as nd

for en in elec_name:
    i = int(en) - 1
    w.set_voltage(en, C[3,i])
    
pot = lambda z : w.compute_total_dc_potential([xp,yp,z])
dpot = nd.Derivative(pot,n=2)
dlist = []
for z_i in xrange(Nz):
    dlist.append(dpot(Z[z_i]))
plot(Z,dlist)


# <codecell>

for en in elec_name:
    i = int(en) - 1
    w.set_voltage(en, C[4,i]-3.1*C[3,i])
    
pot = lambda z : w.compute_total_dc_potential([xp,yp,z])
dpot = nd.Derivative(pot,n=2)
dlist4 = []
for z_i in xrange(Nz):
    dlist4.append(dpot(Z[z_i]))
plot(Z,dlist4)
figure()
pot = lambda z : w.compute_total_dc_potential([xp,yp,z])
dpot = nd.Derivative(pot,n=4)
dlist4 = []
for z_i in xrange(Nz):
    dlist4.append(dpot(Z[z_i]))
plot(Z,dlist4)

# <codecell>

Z1 = linspace(-20e-6,20e-6)
r0 =1.0e6
pot4 = 1*(r0**4)*(35 * (Z1)**4 - 30 * (Z1)**2 / r0**2 + 3 / r0**4)
plot(Z1,pot4)

# <codecell>

Z = linspace(Z_center-200e-6,Z_center+200e-6,Nz)

for en in elec_name:
    i = int(en) 
    w.set_voltage(en, u[i,0])
    
potlist5 = []
for z_i in xrange(Nz):
    r = [xp,yp,Z[z_i]]
    a = w.compute_total_dc_potential(r)
    potlist5.append(a)       
potlist5 = array(potlist5) #/ np.max(np.abs(potlist4))
potlist5 = (potlist5) - np.min((potlist5))

plot(Z, potlist5)
figure()
for en in elec_name:
    i = int(en) 
    w.set_voltage(en, u[i,1])
    
potlist5 = []
for z_i in xrange(Nz):
    r = [xp,yp,Z[z_i]]
    a = w.compute_total_dc_potential(r)
    potlist5.append(a)       
potlist5 = array(potlist5) #/ np.max(np.abs(potlist4))
potlist5 = (potlist5) - np.min((potlist5))

plot(Z, potlist5)

# <codecell>

matshow(abs(u).transpose())
u[1:,0]

# <codecell>

def get_potential(voltages, pos_list, elec_name=elec_name, w=w):
    pos_list = array(pos_list)
    for en in elec_name:
        i = int(en) - 1
        w.set_voltage(en, voltages[i])
    potlist5 = []
    for z_i in xrange(pos_list.shape[0]):
        r = [xp,yp,pos_list[z_i]]
        a = w.compute_total_dc_potential(r)
        potlist5.append(a)       
    potlist5 = array(potlist5)    
    potlist5 = (potlist5) - np.min((potlist5))
    return potlist5
    
def pot_opt(voltages, z_center=1270e-6, n_pos=5, zrange=100e-6, do_plot=True):
    #uc = 20e4
    #ideal_fun = lambda x:uc*(x-z_center)**2
    uc = 20e16
    ustray = 1e5
    ideal_fun = lambda x:uc*(x-z_center)**4 #+ ustray * (x-z_center)
    
    z_pos = linspace(z_center-zrange, z_center+zrange, n_pos)
    potlist = get_potential(voltages,z_pos)
    if do_plot:
        plot(z_pos, ideal_fun(z_pos))
        plot(z_pos, potlist,'o-')
    return sum((ideal_fun(z_pos) - potlist)**2) 

pot_opt(C[3,:],do_plot=True)
    

# <codecell>

from scipy.optimize import minimize
b = minimize(pot_opt, C[3,:])

# <codecell>

bar(arange(23),b.x/1e5)

# <codecell>

bar(arange(23),C[3,:])

# <codecell>

Nz = 10
#Z_center = 760e-6
Z_center = 1270e-6
Z = linspace(Z_center-200e-6,Z_center+200e-6,Nz)

for en in elec_name:
    i = int(en) - 1
    w.set_voltage(en, C[3,i])
    
potlist4 = []
for z_i in xrange(Nz):
    r = [xp,yp,Z[z_i]]
    a = w.compute_total_dc_potential(r)
    potlist4.append(a)       
potlist4 = array(potlist4) #/ np.max(np.abs(potlist4))
potlist4 = (potlist4) - np.min((potlist4))


for en in elec_name:
    i = int(en) - 1
    w.set_voltage(en, b.x[i]/1e5)
    
potlist5 = []
elist = []
for z_i in xrange(Nz):
    print z_i
    r = [xp,yp,Z[z_i]]
    Ex, Ey, Ez = 0,0,0
    for en in elec_name:             
        w.dc_electrode_dict[en].expand_in_multipoles(r, r0=1)
        x = w.dc_electrode_dict[en]
        Ex += x.multipole_dict['Ex']
        Ey += x.multipole_dict['Ey']
        Ez += x.multipole_dict['Ez']
    elist.append([Ex,Ey,Ez])
    a = w.compute_total_dc_potential(r)
    potlist5.append(a)       

elist = array(elist)    
potlist5 = array(potlist5) #/ np.max(np.abs(potlist4))
potlist5 = (potlist5) - np.min((potlist5))

plot(Z,potlist4,'b')
plot(Z,potlist5,'r')
figure()
plot(Z,elist[:,0],'b',label='Ex')
plot(Z,elist[:,1],'r',label='Ey')
plot(Z,elist[:,2],'k',label='Ez')
legend()

# <codecell>

bar(arange(23),b.x)

# <codecell>

xp ,yp

# <codecell>

cfile_index = [0,1,2,4,5]
C1 = zeros((5,23))
C1[0:4,:] = C
C1[4,:] = b.x /1e5
cfile = zeros((8,23))
index1 = 0
for i in cfile_index:
    cfile[i,:] = C1[index1,:]
    index1 += 1
cout = reshape(cfile,(23*8,1))
np.savetxt("Atrap_el5_fourthorder.txt", cout)
cfile[5,:]

# <codecell>


# <codecell>

C[0,:]

# <codecell>

x

# <codecell>

#pot = lambda x: x**2
#x = np.arange(-50, 50, 1)*1e-6
#pot = pot(x)
#plot(np.log(x),np.log(pot))
plot(np.log(zwalk),np.log(pot))

# <codecell>

x=[1,2,3]
print x
x.reverse()
print x

# <codecell>

mult_arr

# <codecell>

w.dc_electrode_dict

# <codecell>


