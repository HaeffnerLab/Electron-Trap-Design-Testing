'''
Module for computing analytic derivatives of rectangles
when the axes_permutation = 0
'''

from sympy import *
import cPickle as pickle
from cloud.serialization.cloudpickle import dump

try:
    functions_dict = pickle.load(open('ad_0.pickle', 'rb'))

except:
    
    print "Did not find precomputed derivatives, so I need to compute them now.\nThis will take a while, but it should be faster next time!"

    x1, x2, y1, y2, x, y, z = symbols('x1 x2 y1 y2 x y z')
    term = ( atan( ((x2 - x)*(y2 - z)) / (y*sqrt( (x2 - x)**2 + (y2 - z)**2 + y**2))) -
             atan( ((x1 - x)*(y2 - z)) / (y*sqrt( (x1 - x)**2 + (y2 - z)**2 + y**2))) -
             atan( ((x2 - x)*(y1 - z)) / (y*sqrt( (x2 - x)**2 + (y1 - z)**2 + y**2))) +
             atan( ((x1 - x)*(y1 - z)) / (y*sqrt( (x1 - x)**2 + (y1 - z)**2 + y**2))) )
    
    syms = [x1, x2, y1, y2, x, y, z]
    functions_dict = {}
    functions_dict['phi0'] = lambdify(syms, term)
    
    # first order
    
    functions_dict['ddx'] = lambdify(syms, term.diff(x))
    functions_dict['ddy'] = lambdify(syms, term.diff(y))
    functions_dict['ddz'] = lambdify(syms, term.diff(z))
    
    # second order
    
    functions_dict['d2dx2'] = lambdify(syms, term.diff(x,2))
    functions_dict['d2dy2'] = lambdify(syms, term.diff(y,2))
    functions_dict['d2dz2'] = lambdify(syms, term.diff(z,2))
    functions_dict['d2dxdy'] = lambdify(syms, term.diff(x,y))
    functions_dict['d2dxdz'] = lambdify(syms, term.diff(x,z))
    functions_dict['d2dydz'] = lambdify(syms, term.diff(y,z))
    
    # third order
    
    functions_dict['d3dz3'] = lambdify(syms, term.diff(z,3))
    functions_dict['d3dxdz2'] = lambdify(syms, term.diff(x,z,2))
    functions_dict['d3dydz2'] = lambdify(syms, term.diff(y,z,2))
    
    # fourth order
    
    functions_dict['d4dz4'] = lambdify(syms, term.diff(z,4))
    functions_dict['d4dx2dz2'] = lambdify(syms, term.diff(x,2,z,2))
    functions_dict['d4dy2dz2'] = lambdify(syms, term.diff(y,2,z,2))

    dump(functions_dict, open('ad_0.pickle', 'wb'))
