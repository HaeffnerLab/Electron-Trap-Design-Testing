import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import time

class trap:
	q = -1.60217662e-19 # coulombs
	m = 9.10938356e-31 #kg (electron)
	#m = 6.6359437706294e-26 #(calcium)
	kB = 1.38064852e-23 # J/K
	f = 1.5e9 # Electrode frequency, in Hertz

	def __init__(self, df, x_max, x_min, y_max, y_min, Nx, Ny, dx, dy):
		self.df = df
		self.x_max = x_max
		self.x_min = x_min
		self.y_max = y_max
		self.y_min = y_min
		self.Nx = Nx
		self.Ny = Ny
		self.dx = dx
		self.dy = dy

	def get_row_index(self, x, y):
	    # given spatial coordinates x and y, 
	    # output the index of the row corresponding to the coordinates
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
	    # return the gradient of x component of electric field at position (x, y);
	    # used to interpolate E field between grid points
	    # note that x, y are supposed to be on grid intersections
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
	    # return the gradient of x component of electric field at position (x, y);
	    # used to interpolate E field between grid points
	    # note that x, y are supposed to be on grid intersections
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
	    # return the electric field at position (x, y) at time t
	    n = self.get_row_index(x, y)
	    x0, y0 = self.df.iloc[n, 0], self.df.iloc[n, 1]
	    Ex0, Ey0 = self.df.iloc[n, 3], self.df.iloc[n, 4]
	    Ex = Ex0 + self.grad_Ex(x0, y0)[0] * (x-x0) + self.grad_Ex(x0, y0)[1] * (y-y0)
	    Ey = Ey0 + self.grad_Ey(x0, y0)[0] * (x-x0) + self.grad_Ey(x0, y0)[1] * (y-y0)
	    return (Ex*np.cos(2*np.pi*self.f*t), Ey*np.cos(2*np.pi*self.f*t))

	def acceleration(self, x, y, t):
	    Ex, Ey= self.E_field(x, y, t)
	    return np.array([Ex*self.q/self.m, Ey*self.q/self.m])

	def within_boundary(self, x, y):
	    # check whether the x, y, z coordinate of an electron is within
	    # our area of interest.
	    n = self.get_row_index(x, y)
	    #return x<x_max and x>x_min and y<y_max and y>y_min
	    return x<self.x_max and x>self.x_min and y<self.y_max and y>self.y_min and \
	(abs(self.df.iloc[n, 3]) >= 1.0e-7 or abs(self.df.iloc[n, 4]) >= 1.0e-7) 

	def rk4(self, y, time, dt, derivs): 
	    f0 = derivs(y, time)
	    fhalf_bar = derivs(y+f0*dt/2, time+dt) 
	    fhalf = derivs(y+fhalf_bar*dt/2, time+dt)
	    f1_bar= derivs(y+fhalf*dt, time+dt)
	    y_next = y+dt/6*(f0+2*fhalf_bar+2*fhalf+f1_bar)
	    return y_next

	def E_field_sim(self, state, time):
	    # this is a function that takes in a 2-element array state and a number time.
	    # state[0] is current position and state[1] is current velocity.
	    # it calculates the derivative of state, and returns a 2-element array,
	    # with the 0th element being derivative of position and the 1th being 
	    # the acceleration
	    g0 = state[1]
	    x, y = state[0]
	    g1 = self.acceleration(x, y, time)
	    return np.array([g0, g1])

	def trapped(self, rou, phi, v, theta, dt, t_max):

	    electron_pos=np.array([rou*np.cos(phi), rou*np.sin(phi)])
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

	def traj(self, rou, phi, v, theta, dt, t_max):
		# Input the initial condition, output a list of three lists,
		# the first one is a list of time stamps, the second and third
		# are lists of x, y coordinates of the particle at the time stamps
	    electron_pos=np.array([rou*np.cos(phi), rou*np.sin(phi)])
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
	    # Simulate ion in the trap with initial condition characterized by a 
	    # Boltzmann distribution of temperature T. The simulation (using RK4) step size 
	    # is dt, and the maximum time duration of the simulation is t_max. The initial position
	    # of the ion is characterized by a Gaussian distribution with full-width-half max FWHM.
	    # the function returns a list of length N_ion_sample. 
	    # The ith element of the list is 1 iff the ith ion is successfully trapped and is 0 if 
	    # the ion escaped.
	    class Boltzmann(stats.rv_continuous):
	        def _pdf(self, v):
	            return m*v*np.exp((-1/2*m*v**2)/(kB*T))/(kB*T)
	    result = []
	    Boltzmann_dist = Boltzmann(a=0)
	    for i in range(N_ion_samples):
	        rou = abs(np.random.normal(0, FWHM/2.355))
	        phi = np.random.uniform(0, np.pi*2)
	        v = Boltzmann_dist.rvs()
	        theta = np.random.uniform(0, np.pi*2)
	        if self.trapped(rou, phi, v, theta, dt, t_max):
	            result.append(1)
	        else:
	            result.append(0)
	        if i % 20 == 0:
	            print(i+1, " ion samples already simulated")
	    print("T: ", T, "Kelvin")
	    print("N_samples: ", N_ion_samples)
	    print("Time step: ", dt*1.0e9, "ns")
	    print("Max Sim Time: ", t_max*1.0e6, "us")
	    print("FWHM: ", FWHM*1.0e6, "um")
	    print("Trapping Rate: ", np.mean(result), "+/-", np.std(result)/np.sqrt(len(result)))
	    print("------")
	    return result


class quarter_trap(trap):
	
	def __init__(self, df, x_max, x_min, y_max, y_min, Nx, Ny, dx, dy, f=trap.f):
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

	def get_row_index(self, x, y):
	    # given spatial coordinates x and y, 
	    # output the index of the row corresponding to the coordinates
	    # this only works for the first quadrant
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
	    # return the gradient of x component of electric field at position (x, y);
	    # used to interpolate E field between grid points
	    # note that x, y are supposed to be on grid intersections
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
	    # return the gradient of x component of electric field at position (x, y);
	    # used to interpolate E field between grid points
	    # note that x, y are supposed to be on grid intersections
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
	    # return the electric field at position (x, y) at time t
	    x, y = abs(x), abs(y)
	    n = self.get_row_index(x, y)
	    x0, y0 = self.df.iloc[n, 0], self.df.iloc[n, 1]
	    Ex0, Ey0 = self.df.iloc[n, 3], self.df.iloc[n, 4]
	    Ex = Ex0 + self.grad_Ex(x0, y0)[0] * (x-x0) + self.grad_Ex(x0, y0)[1] * (y-y0)
	    Ey = Ey0 + self.grad_Ey(x0, y0)[0] * (x-x0) + self.grad_Ey(x0, y0)[1] * (y-y0)
	    return (Ex*np.cos(2*np.pi*self.f*t), Ey*np.cos(2*np.pi*self.f*t))

	def within_boundary(self, x, y):
	    # check whether the x, y, z coordinate of an electron is within
	    # our area of interest.
	    x, y = abs(x), abs(y)
	    n = self.get_row_index(x, y)
	    #return x<x_max and x>x_min and y<y_max and y>y_min
	    return x<self.x_max and y<self.y_max and \
	(abs(self.df.iloc[n, 3]) >= 1.0e-7 or abs(self.df.iloc[n, 4]) >= 1.0e-7) 


class half_trap(trap):
	
	def __init__(self, df, x_max, x_min, y_max, y_min, Nx, Ny, dx, dy, f=trap.f):
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

	def get_row_index(self, x, y):
	    # given spatial coordinates x and y, 
	    # output the index of the row corresponding to the coordinates
	    # this only works for the first quadrant
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
	    # return the gradient of x component of electric field at position (x, y);
	    # used to interpolate E field between grid points
	    # note that x, y are supposed to be on grid intersections
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
	    # return the gradient of x component of electric field at position (x, y);
	    # used to interpolate E field between grid points
	    # note that x, y are supposed to be on grid intersections
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
	    # return the electric field at position (x, y) at time t
	    x = abs(x)
	    n = self.get_row_index(x, y)
	    x0, y0 = self.df.iloc[n, 0], self.df.iloc[n, 1]
	    Ex0, Ey0 = self.df.iloc[n, 3], self.df.iloc[n, 4]
	    Ex = Ex0 + self.grad_Ex(x0, y0)[0] * (x-x0) + self.grad_Ex(x0, y0)[1] * (y-y0)
	    Ey = Ey0 + self.grad_Ey(x0, y0)[0] * (x-x0) + self.grad_Ey(x0, y0)[1] * (y-y0)
	    return (Ex*np.cos(2*np.pi*self.f*t), Ey*np.cos(2*np.pi*self.f*t))


	def within_boundary(self, x, y):
	    # check whether the x, y, z coordinate of an electron is within
	    # our area of interest.
	    x = abs(x)
	    n = self.get_row_index(x, y)
	    #return x<x_max and x>x_min and y<y_max and y>y_min
	    return x<self.x_max and y<self.y_max and y>self.y_min and \
	(abs(self.df.iloc[n, 3]) >= 1.0e-7 or abs(self.df.iloc[n, 4]) >= 1.0e-7) 


	def hit_electrodes(self, x, y):
		x = abs(x)
		n = self.get_row_index(x, y)
		return (abs(self.df.iloc[n, 3]) <= 1.0e-7 or\
			abs(self.df.iloc[n, 4]) <= 1.0e-7) 

	