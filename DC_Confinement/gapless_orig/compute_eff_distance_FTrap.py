from FTrap import FTrapWorld, centerxRange, centeryRange, u

"""  For a given electrode number use the gapless approximation to estimate the effective coupling 
      distance at the trapping position
	  
	   FTrapWorld = a World Object containing the F trap electrodes. the coordinate origin is on the 
	    trap axis, between the lower edges of the two lowermost electrodes, 1 and 11"""
		
import numpy as np
import matplotlib.pyplot as plt
def plotRF():
    """plot the RF amplitude"""
    xrng = np.linspace(centerxRange[0], centerxRange[1], 10)
    yrng = np.linspace(centeryRange[0], centeryRange[1], 10)
    print xrng
    print yrng
    [X, Y] = np.meshgrid(xrng, yrng, indexing='ij')
    z = 10*u
    xlist = np.ravel(X)
    ylist = np.ravel(Y)
    V = np.empty(xlist.shape)

    FTrapWorld.set_omega_rf(30E6)

    for i in range(len(xlist)):
        print i
        r = (xlist[i], ylist[i], z)
        print r
        V[i] = FTrapWorld.compute_pseudopot(r)
		
    
    V = V.reshape(X.shape)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, V)
    plt.show()
	

def getMinimumPseudopot(N):
    """ return approximate minimum of the trap pseudopotential
	     in the dumbest way possible
		 sample N^3 points in a cube about the geometric trap center.
		 the saddle point is chosen as that with the minimum magnitude of
		 gradient"""

    x0 = 0.5 * (centerxRange[0] + centerxRange[1])
    y0 = 0.5 * (centeryRange[0] + centeryRange[1])
	
    z0 = 200*u
	
    
    width = 2*z0
    gv = np.linspace(-width/2, width/2, N)
    xrng = gv.copy() + x0
    yrng = gv.copy() + y0
    zrng = gv.copy() + z0
    
   
    V = np.empty([N, N, N])    
    
    for i in range(N):
        for j in range(N):
            for k in range(N):
                #print i, j, k
                r = (xrng[i], yrng[j], zrng[k])
                V[i,j,k] = FTrapWorld.compute_pseudopot(r)

    dV = np.abs(np.gradient(V))
    dVmin = np.amin(dV)
    mn = np.where(dV == dVmin)	
    imin = mn[0][0]
    jmin = mn[1][0]
    kmin = mn[2][0]
    rmin = (xrng[imin], yrng[jmin], zrng[kmin])
				
    return rmin

#electrode number
plt.ion()
plotField = 1
def plotCouplingStrength(name):
    el = FTrapWorld.dc_electrode_dict[name]
	
	#1V on target electrode, all else zero
    for nm in FTrapWorld.dc_electrode_dict.keys(): 
        otherEl = FTrapWorld.dc_electrode_dict[nm]
        print nm, name
        if (nm == name):
            otherEl.set_voltage(1.0)
        else:
            otherEl.set_voltage(0.0)
        #print "electrode {0} is at {1}".format(nm, FTrapWorld.dc_electrode_dict[nm].voltage)
    #xp = 0
    yp = 0.5 * (centeryRange[1] - centeryRange[0])
    #yp= 1215*u
    xp = 0*u
	
    z = np.linspace(0, 200*u, 100)
    dx = np.empty(np.size(z))
    dy = np.empty(np.size(z))
    dz = np.empty(np.size(z))
	
    for i in range(len(dx)):
        #print i
        print "using trap location {0}, {1}, {2} m".format(xp, yp, z[i])
        xd, yd, zd = el.compute_d_effective((xp, yp, z[i]))
        dx[i] = xd
        dy[i] = yd
        dz[i] = zd
	print "Electrode bounding box: x:({0}, {1})   y:({2}, {3})".format(el.x1, el.x2, el.y1, el.y2)
    plt.figure()
    plt.grid(True)
    plt.plot(  z/u, dz)		
    plt.xlabel('height, um')
    plt.ylabel('deff, m')
    plt.legend(['dz'])
    plt.title(name)
	
    z0 = 100*u
	
    if plotField:
        plt.figure()
        Ez = 1.0/dz
        plt.plot(z/u, Ez)
        plt.xlabel('height, um')
        plt.ylabel('field strength V/m')
        plt.grid(True)
        plt.title('Ez at ion position, electrode {0}'.format(name))

        xg = np.linspace(xp-50*u, xp+ 50*u, 20)
        yg = np.linspace(yp-50*u, yp+50*u, 20)
        X, Y = np.meshgrid(xg, yg)
        xlist, ylist = np.ravel(X), np.ravel(Y)		
        Ez = np.empty(len(xlist))
        
        for i in range(len(xlist)):
            Ez[i] = el.compute_electric_field((xlist[i], ylist[i], z0))[2]
        Ez = Ez.reshape(X.shape)
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Ez)
        plt.show()
    

    

#FTrapWorld.set_omega_rf(30E6)
#FTrapWorld.set_voltage('RF1', 1E4)
#FTrapWorld.set_voltage('RF2', 1E4)
#rmin = getMinimumPseudopot(50)	
#for i in range(1, 10):#
#    nm = str(i)
#    plotCouplingStrength(nm)
#plotRF()	
plotCouplingStrength('21')