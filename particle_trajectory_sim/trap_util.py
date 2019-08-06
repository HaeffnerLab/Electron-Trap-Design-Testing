import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import time

# Defining physical constants for electron trap. 
# Remember to change q and m when you change particle type.
q = -1.60217662e-19 # coulombs
m = 9.10938356e-31 #kg (electron)
#m = 6.6359437706294e-26 #(calcium)
kB = 1.38064852e-23 # J/K

class trap:
	"""
	A class used to represent a single particle (electron or ion) trap.
	
	Attributes
	----------
	df: pandas data frame
		the data frame that stores the E-field data
	x_max, x_min, y_max, y_min: floats
		the boundaries of the region this code considers for this trap
	Nx, Ny: integers
		the number of grid points along each dimension
	dx, dy: floats
		spatial step between grids along each dimension
	f: float
		driving frequency of the trap
	q: float
		charge of the trapped particle
	m: float
		mass of the trapped particle

	Methods that are frequently used 
	----------
	extracted(rho, phi, v, theta, dt, t_max)
		(for particle ejection potential) for a given initial condition and 
		maximum simulation time, calculate whether the particle will be 
		successfully extracted towards the positive y direction
	traj(self, rho, phi, v, theta, dt, t_max)
		get the trajectory of a single particle simulation with given initial condition
	Boltzmann_sim(self, N_ion_samples, T, dt, t_max, FWHM)
		perform simulation many times under a fixed temperature with the initial 
		condition given by Boltzmann distribution
	"""

	def __init__(self, df, x_max, x_min, y_max, y_min, Nx, Ny, dx, dy, f, q=q, m=m):
		"""Initializing the trap object. This code reads from a dataframe of 6 columns:
		x, y, z, Ex, Ey, Ez. However this code only deals with 2D trajectory in the 
		x-y plane, but it should be fairly easy to modify it to 3D.

		Parameters:
		----------
		df: pandas data frame
			the data frame that stores the E-field data; each row of the dataframe
			corresponds to the E-field data of a specific point in space; the rows 
			should be arranged and indexed such that going down the dataframe 
			with increasing indices, y changes first and then x changes.
		x_max, x_min, y_max, y_min: floats
			the boundaries of the region this code considers for this trap
		Nx, Ny: integers
			the number of grid points along each dimension
		dx, dy: floats
			spatial step between grids along each dimension
		f: float
			driving frequency of the trap; for DC field, set f to 0
		q: float
			charge of the trapped particle
		m: float
			mass of the trapped particle
		"""

		self.df = df
		self.x_max = x_max
		self.x_min = x_min
		self.y_max = y_max
		self.y_min = y_min
		self.Nx = Nx
		self.Ny = Ny
		self.dx = dx
		self.dy = dy
		self.f = f
		self.q = q
		self.m = m
		self.kB = kB

	def get_row_index(self, x, y):
	    """ Given spatial coordinates x and y, return the dataframe roew index of 
	    the corresponding to the coordinates"""
	    if x > self.x_max:
	        x = self.x_max
	    if x < self.x_min:
	        x = self.x_min
	    if y > self.y_max:
	        y = self.y_max
	    if y < self.y_min:
	        y = self.y_min
	    i = int((x - self.x_min) / self.dx)
	    j = int((y - self.y_min) / self.dy)
	    return i * (self.Ny + 1) + j

	def grad_Ex(self, x, y):
	    """Returns the gradient of x component of electric field at position (x, y);
	    this method is used to interpolate E field between grid points;
	    note that x, y are supposed to be on grid intersections
	    """
	    n = self.get_row_index(x, y)
	    if x == self.x_max:
	        nx = self.get_row_index(x-self.dx, y)
	        x_deriv = (self.df.iloc[n,3] - self.df.iloc[nx,3])/(self.dx)
	    else:
	        nx = self.get_row_index(x+self.dx, y)
	        x_deriv = (self.df.iloc[nx,3] - self.df.iloc[n,3])/(self.dx)
	    if y == self.y_max:
	        ny = self.get_row_index(x, y-self.dy)
	        y_deriv = (self.df.iloc[n,3] - self.df.iloc[ny,3])/(self.dy)
	    else:
	        ny = self.get_row_index(x, y+self.dy)
	        y_deriv = (self.df.iloc[ny,3] - self.df.iloc[n,3])/(self.dy)
	    return (x_deriv, y_deriv)

	def grad_Ey(self, x, y):
	    """Returns the gradient of x component of electric field at position (x, y);
	    used to interpolate E field between grid points;
	    note that x, y are supposed to be on grid intersections
	    """
	    n = self.get_row_index(x, y)
	    if x == self.x_max:
	        nx = self.get_row_index(x-self.dx, y)
	        x_deriv = (self.df.iloc[n,4] - self.df.iloc[nx,4])/(self.dx)
	    else:
	        nx = self.get_row_index(x+self.dx, y)
	        x_deriv = (self.df.iloc[nx,4] - self.df.iloc[n,4])/(self.dx)
	    if y == self.y_max:
	        ny = self.get_row_index(x, y-self.dy)
	        y_deriv = (self.df.iloc[n,4] - self.df.iloc[ny,4])/(self.dy)
	    else:
	        ny = self.get_row_index(x, y+self.dy)
	        y_deriv = (self.df.iloc[ny,4] - self.df.iloc[n,4])/(self.dy)
	    return (x_deriv, y_deriv)


	def E_field(self, x, y, t):
	    """Returns the electric field at position (x, y) at time t; the x, y coordinates
	    don't need to fall on an exact grid point (it can be between two grid points)
	    """
	    n = self.get_row_index(x, y)
	    x0, y0 = self.df.iloc[n, 0], self.df.iloc[n, 1]
	    Ex0, Ey0 = self.df.iloc[n, 3], self.df.iloc[n, 4]
	    Ex = Ex0 + self.grad_Ex(x0, y0)[0] * (x-x0) + self.grad_Ex(x0, y0)[1] * (y-y0)
	    Ey = Ey0 + self.grad_Ey(x0, y0)[0] * (x-x0) + self.grad_Ey(x0, y0)[1] * (y-y0)
	    return (Ex*np.cos(2*np.pi*self.f*t), Ey*np.cos(2*np.pi*self.f*t))

	def acceleration(self, x, y, t):
		"""Returns the acceleration of the particle at position (x, y) at time t;
		the x, y coordinates don't need to fall on an exact grid point
		"""
		Ex, Ey= self.E_field(x, y, t)
		return np.array([Ex*self.q/self.m, Ey*self.q/self.m])

	def within_boundary(self, x, y):
	    """Checks whether the x, y, z coordinate is within our area of interest
	    (defined by x_max, x_min, y_max, y_min)
	    """
	    n = self.get_row_index(x, y)
	    #return x<x_max and x>x_min and y<y_max and y>y_min
	    return x<self.x_max and x>self.x_min and y<self.y_max and y>self.y_min and \
	(abs(self.df.iloc[n, 3]) >= 1.0e-7 or abs(self.df.iloc[n, 4]) >= 1.0e-7) 

	def hit_electrodes(self, x, y, E_threshold=1.0e-7):
		"""Checks whether a particle hits an electrode if it is at position (x, y);
		here electrodes are defined as areas where E field is smaller than the 
		E_threshold (default is 1.0e-7)
		"""
		n = self.get_row_index(x, y)
		return (abs(self.df.iloc[n, 3])**2 + abs(self.df.iloc[n, 4])**2 \
			<= E_threshold**2)

	def rk4(self, y, time, dt, derivs): 
		"""Solving an ODE and calculate the value for next step using RK4

		Parameters
		----------
		y: float or 1D array
			the current state of the particle
		time: float
			the current time
		dt: float
			time step for solving the differential equation
		derivs: function
			a function that takes in two arguments: current state and current time;
			it returns the time derivatives of the current state at current time

		Return
		----------
		y_next: float or 1D array
			the state of the particle one time step later from y;
		"""
		f0 = derivs(y, time)
		fhalf_bar = derivs(y+f0*dt/2, time+dt)
		fhalf = derivs(y+fhalf_bar*dt/2, time+dt)
		f1_bar= derivs(y+fhalf*dt, time+dt)
		y_next = y+dt/6*(f0+2*fhalf_bar+2*fhalf+f1_bar)
		return y_next

	def E_field_sim(self, state, time):
	    """This is a function that takes in a 2-element array state and a number time and
	    calculates the time derivatives of the state.

	    Parameters
	    ----------
	    state: 1D array of length 2
	    	state[0] is the current position of the particle and state[1] is 
	    	the current velocity
	    time: 
	    	the current time
	    
	    Return
	    ----------
	    A 1D array with the 0th element being the current velocity and the 1st element being
	    the current acceleration
	    """
	    g0 = state[1]
	    x, y = state[0]
	    g1 = self.acceleration(x, y, time)
	    return np.array([g0, g1])

	def trapped(self, rho, phi, v, theta, dt, t_max):
		"""Calculates whether the particle will be trapped in the trap for at least t_max time.

		Parameters
		---------
		rho: float
			initial distance of the particle from the trap center
		phi: float
			angle between the initial position vector and the positive x-axis
		v: float
			initial speed of the particle
		theta: float
			angle between the initial velocity vector and the positive x axis
		dt: float
			time step of the simulation
		t_max: float
			maximum simulation time

		Return
		----------
		True iff the particle is still in the trapping region after t_max.
		"""
		electron_pos=np.array([rho*np.cos(phi), rho*np.sin(phi)])
		electron_vel = np.array([v*np.cos(theta), v*np.sin(theta)])
		state = np.array([electron_pos, electron_vel])
		t = 0.0 # the time variable
		trapped = True
		# actual simulation
		while t < t_max:
			x, y = state[0]
			if not self.within_boundary(x, y):
				trapped = False
				break
			state = self.rk4(state, t, dt, self.E_field_sim)
			t += dt
		return trapped

	def extracted(self, rho, phi, v, theta, dt, t_max):
	    """Calculates whether the particle will be successfully extracted through 
	    the positive y direction.

		Parameters
		---------
		rho: float
			initial distance of the particle from the trap center
		phi: float
			angle between the initial position vector and the positive x-axis
		v: float
			initial speed of the particle
		theta: float
			angle between the initial velocity vector and the positive x axis
		dt: float
			time step of the simulation
		t_max: float
			maximum simulation time

		Return
		----------
		True iff the particle is extracted through the positive y direction. If the particle
		hits an electrode, or if it escapes through a different direction, or if it still stays 
		in the trap after t_max, the particle has NOT been extracted successfully and the 
		function returns False.
		"""
	    electron_pos=np.array([rho*np.cos(phi), rho*np.sin(phi)])
	    electron_vel = np.array([v*np.cos(theta), v*np.sin(theta)])
	    state = np.array([electron_pos, electron_vel])
	    t = 0.0 # the time variable
	    extracted = False
	    # actual simulation
	    while t < t_max:
	        x, y = state[0]
	        if self.hit_electrodes(x, y):
	            break
	        if y > self.y_max:
	            extracted = True
	            break
	        if not self.within_boundary(x, y):
	            break
	        state = self.rk4(state, t, dt, self.E_field_sim)
	        t += dt
	    return extracted

	def traj(self, rho, phi, v, theta, dt, t_max):
		"""Calculates the particle trajectory of a single particle given certain
		initial condition.

		Parameters
		---------
		rho: float
			initial distance of the particle from the trap center
		phi: float
			angle between the initial position vector and the positive x-axis
		v: float
			initial speed of the particle
		theta: float
			angle between the initial velocity vector and the positive x axis
		dt: float
			time step of the simulation
		t_max: float
			maximum simulation time

		Returns
		----------
		A list of three lists: t_s, x_traj, y_traj. x_traj[i] and y_traj[i] correspond
		to the (x, y) coordinates of the particle at time t_s[i]
		"""
		electron_pos=np.array([rho*np.cos(phi), rho*np.sin(phi)])
		electron_vel = np.array([v*np.cos(theta), v*np.sin(theta)])
		state = np.array([electron_pos, electron_vel])
		t = 0.0 # the time variable
		trapped = True
		x_traj, y_traj, t_s = [], [], []
		# actual simulation
		while t < t_max:
			x, y = state[0]
			if not self.within_boundary(x, y):
				trapped = False
				print("Out of bound")
				break
			x_traj.append(x)
			y_traj.append(y)
			t_s.append(t)
			state = self.rk4(state, t, dt, self.E_field_sim)
			t += dt
		return [t_s, x_traj, y_traj]


	def Boltzmann_sim(self, N_ion_samples, T, dt, t_max, FWHM):
	    """Simulate a certain number of particles one by one, each having an initial 
	    velocity characterized by a Boltzmann distribution of temperature T and 
	    initial position distribution characterized by a Gaussian with full width 
	    at half maximum equals FWHM. 

		Parameters
		----------
	    N_ion_samples: int
	    	the number of ion samples that get simulated; the function will simulate
	    	this number of ions one by one and returns the results of all of them
	    T: float
	    	temperature
	    dt: float
	    	time step of the simulation
	    t_max: float
	    	maximum simulation time
	    FWHM: float
	    	full width at half maximum of the Gaussian distribution that describes
	    	the initial position of the particle
	    
	    Returns
	    ---------
	    result: list
	    	the ith element of the list is 1 iff the ith ion is successfully trapped 
	    	and is 0 if the ion escaped.
	    """
	    m = self.m
	    kB = self.kB
	    class Boltzmann(stats.rv_continuous):
	        def _pdf(self, v):
	            return m*v*np.exp((-1/2*m*v**2)/(kB*T))/(kB*T)
	    result = []
	    Boltzmann_dist = Boltzmann(a=0)
	    for i in range(N_ion_samples):
	        rho = abs(np.random.normal(0, FWHM/2.355))
	        phi = np.random.uniform(0, np.pi*2)
	        v = Boltzmann_dist.rvs()
	        theta = np.random.uniform(0, np.pi*2)
	        if self.trapped(rho, phi, v, theta, dt, t_max):
	            result.append(1)
	        else:
	            result.append(0)
	        if i % 20 == 0:
	            print(i+1, " samples already simulated")
	    print("T: ", T, "Kelvin")
	    print("N_samples: ", N_ion_samples)
	    print("Time step: ", dt*1.0e9, "ns")
	    print("Max Sim Time: ", t_max*1.0e6, "us")
	    print("FWHM: ", FWHM*1.0e6, "um")
	    print("Trapping Rate: ", np.mean(result), "+/-", np.std(result)/np.sqrt(len(result)))
	    print("------")
	    return result


class quarter_trap(trap):
	"""a subclass of class trap, where instead of taking in a dataframe consisting of 
	the E-field data of the whole trapping region, it only takes in a dataframe consisting
	of the E-field data of the first quadrant of the plane; needs to infer the E-field 
	data elsewhere by symmetry"""

	def __init__(self, df, x_max, x_min, y_max, y_min, Nx, Ny, dx, dy, f, q=q, m=m):
		self.df = df
		self.x_max = x_max
		self.x_min = x_min
		self.y_max = y_max
		self.y_min = y_min
		self.Nx = Nx
		self.Ny = Ny
		self.dx = dx
		self.dy = dy
		self.f = f
		self.q = q
		self.m = m
		self.kB = kB

	def get_row_index(self, x, y):
	   	"""for documentation, see the method with the same name under class trap"""
	   	if x > self.x_max:
	   		x = self.x_max
	   	if x < self.x_min:
	   		x = self.x_min
	   	if y > self.y_max:
	   		y = self.y_max
	   	if y < self.y_min:
	   		y = self.y_min
	   	i = int((x - self.x_min) / self.dx)
	   	j = int((y - self.y_min) / self.dy)
	   	return i * (self.Ny + 1) + j

	def grad_Ex(self, x, y):
	    """for documentation, see the method with the same name under class trap"""
	    x, y = abs(x), abs(y)
	    n = self.get_row_index(x, y)
	    if x == self.x_max:
	        nx = self.get_row_index(x-self.dx, y)
	        x_deriv = (self.df.iloc[n,3] - self.df.iloc[nx,3])/(self.dx)
	    else:
	        nx = self.get_row_index(x+self.dx, y)
	        x_deriv = (self.df.iloc[nx,3] - self.df.iloc[n,3])/(self.dx)
	    if y == self.y_max:
	        ny = self.get_row_index(x, y-self.dy)
	        y_deriv = (self.df.iloc[n,3] - self.df.iloc[ny,3])/(self.dy)
	    else:
	        ny = self.get_row_index(x, y+self.dy)
	        y_deriv = (self.df.iloc[ny,3] - self.df.iloc[n,3])/(self.dy)
	    return (x_deriv, y_deriv)

	def grad_Ey(self, x, y):
	    """for documentation, see the method with the same name under class trap"""
	    x, y = abs(x), abs(y)
	    n = self.get_row_index(x, y)
	    if x == self.x_max:
	        nx = self.get_row_index(x-self.dx, y)
	        x_deriv = (self.df.iloc[n,4] - self.df.iloc[nx,4])/(self.dx)
	    else:
	        nx = self.get_row_index(x+self.dx, y)
	        x_deriv = (self.df.iloc[nx,4] - self.df.iloc[n,4])/(self.dx)
	    if y == self.y_max:
	        ny = self.get_row_index(x, y-self.dy)
	        y_deriv = (self.df.iloc[n,4] - self.df.iloc[ny,4])/(self.dy)
	    else:
	        ny = self.get_row_index(x, y+self.dy)
	        y_deriv = (self.df.iloc[ny,4] - self.df.iloc[n,4])/(self.dy)
	    return (x_deriv, y_deriv)


	def E_field(self, x, y, t):
	    """for documentation, see the method with the same name under class trap"""
	    x, y = abs(x), abs(y)
	    n = self.get_row_index(x, y)
	    x0, y0 = self.df.iloc[n, 0], self.df.iloc[n, 1]
	    Ex0, Ey0 = self.df.iloc[n, 3], self.df.iloc[n, 4]
	    Ex = Ex0 + self.grad_Ex(x0, y0)[0] * (x-x0) + self.grad_Ex(x0, y0)[1] * (y-y0)
	    Ey = Ey0 + self.grad_Ey(x0, y0)[0] * (x-x0) + self.grad_Ey(x0, y0)[1] * (y-y0)
	    return (Ex*np.cos(2*np.pi*self.f*t), Ey*np.cos(2*np.pi*self.f*t))

	def within_boundary(self, x, y):
	    """for documentation, see the method with the same name under class trap"""
	    x, y = abs(x), abs(y)
	    n = self.get_row_index(x, y)
	    #return x<x_max and x>x_min and y<y_max and y>y_min
	    return x<self.x_max and y<self.y_max and \
	(abs(self.df.iloc[n, 3]) >= 1.0e-7 or abs(self.df.iloc[n, 4]) >= 1.0e-7) 

	def hit_electrodes(self, x, y, E_threshold=1.0e-7):
		"""for documentation, see the method with the same name under class trap"""
		x, y = abs(x), abs(y)
		n = self.get_row_index(x, y)
		return (abs(self.df.iloc[n, 3]) <= E_threshold and\
			abs(self.df.iloc[n, 4]) <= E_threshold)


class half_trap(trap):
	"""a subclass of class trap, where instead of taking in a dataframe consisting of 
	the E-field data of the whole trapping region, it only takes in a dataframe consisting
	of the E-field data of the positive x plane; needs to infer the E-field data elsewhere
	by symmetry"""

	def __init__(self, df, x_max, x_min, y_max, y_min, Nx, Ny, dx, dy, f, q=q, m=m):
		self.df = df
		self.x_max = x_max
		self.x_min = x_min
		self.y_max = y_max
		self.y_min = y_min
		self.Nx = Nx
		self.Ny = Ny
		self.dx = dx
		self.dy = dy
		self.f = f
		self.q = q
		self.m = m
		self.kB = kB

	def get_row_index(self, x, y):
	    """for documentation, see the method with the same name under class trap"""
	    if x > self.x_max:
	        x = self.x_max
	    if x < self.x_min:
	        x = self.x_min
	    if y > self.y_max:
	        y = self.y_max
	    if y < self.y_min:
	        y = self.y_min
	    i = int((x - self.x_min) / self.dx)
	    j = int((y - self.y_min) / self.dy)
	    return i * (self.Ny + 1) + j

	def grad_Ex(self, x, y):
	    """for documentation, see the method with the same name under class trap"""
	    x = abs(x)
	    n = self.get_row_index(x, y)
	    if x == self.x_max:
	        nx = self.get_row_index(x-self.dx, y)
	        x_deriv = (self.df.iloc[n,3] - self.df.iloc[nx,3])/(self.dx)
	    else:
	        nx = self.get_row_index(x+self.dx, y)
	        x_deriv = (self.df.iloc[nx,3] - self.df.iloc[n,3])/(self.dx)
	    if y == self.y_max:
	        ny = self.get_row_index(x, y-self.dy)
	        y_deriv = (self.df.iloc[n,3] - self.df.iloc[ny,3])/(self.dy)
	    else:
	        ny = self.get_row_index(x, y+self.dy)
	        y_deriv = (self.df.iloc[ny,3] - self.df.iloc[n,3])/(self.dy)
	    return (x_deriv, y_deriv)

	def grad_Ey(self, x, y):
	    """for documentation, see the method with the same name under class trap"""
	    x = abs(x)
	    n = self.get_row_index(x, y)
	    if x == self.x_max:
	        nx = self.get_row_index(x-self.dx, y)
	        x_deriv = (self.df.iloc[n,4] - self.df.iloc[nx,4])/(self.dx)
	    else:
	        nx = self.get_row_index(x+self.dx, y)
	        x_deriv = (self.df.iloc[nx,4] - self.df.iloc[n,4])/(self.dx)
	    if y == self.y_max:
	        ny = self.get_row_index(x, y-self.dy)
	        y_deriv = (self.df.iloc[n,4] - self.df.iloc[ny,4])/(self.dy)
	    else:
	        ny = self.get_row_index(x, y+self.dy)
	        y_deriv = (self.df.iloc[ny,4] - self.df.iloc[n,4])/(self.dy)
	    return (x_deriv, y_deriv)


	def E_field(self, x, y, t):
	    """for documentation, see the method with the same name under class trap"""
	    x = abs(x)
	    n = self.get_row_index(x, y)
	    x0, y0 = self.df.iloc[n, 0], self.df.iloc[n, 1]
	    Ex0, Ey0 = self.df.iloc[n, 3], self.df.iloc[n, 4]
	    Ex = Ex0 + self.grad_Ex(x0, y0)[0] * (x-x0) + self.grad_Ex(x0, y0)[1] * (y-y0)
	    Ey = Ey0 + self.grad_Ey(x0, y0)[0] * (x-x0) + self.grad_Ey(x0, y0)[1] * (y-y0)
	    return (Ex*np.cos(2*np.pi*self.f*t), Ey*np.cos(2*np.pi*self.f*t))


	def within_boundary(self, x, y):
	    """for documentation, see the method with the same name under class trap"""
	    x = abs(x)
	    n = self.get_row_index(x, y)
	    #return x<x_max and x>x_min and y<y_max and y>y_min
	    return x<self.x_max and y<self.y_max and y>self.y_min and \
	(abs(self.df.iloc[n, 3]) >= 1.0e-7 or abs(self.df.iloc[n, 4]) >= 1.0e-7) 


	def hit_electrodes(self, x, y, E_threshold=1.0e-7):
		"""for documentation, see the method with the same name under class trap"""
		x = abs(x)
		n = self.get_row_index(x, y)
		return (abs(self.df.iloc[n, 3]) <= E_threshold and\
			abs(self.df.iloc[n, 4]) <= E_threshold) 

	