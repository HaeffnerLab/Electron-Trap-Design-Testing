'''
Tool for mapping stray fields along the axial direction
'''

import numpy as np
from labrad import units as u

class FieldMap():
    
    def axial_field_map(self, U2, ion_positions, axial_offset = 0):
        '''
        Takes the inteded axial curvature and a numpy array of ion positions.
        Returns the electric field in the axial direction at each ion position

        U2 is intended in V/m^2, ion_positions in meters

        Assumes coordinates in which the center of the applied field is
        at x = 0
        '''

        q = u.e['C']
        e0 = u.eps0['C/V/m']
        k = q/(4*np.pi*e0) # fields will be in V/m

        ion_fields = [] # electric field due to the other ions
        for p in ion_positions:
            Ep = 0
            for d in set(ion_positions).difference({p}): # loop over the other ion positions
                r = p - d # positive whenver d < p, to make positive field at the ion position
                Ep += np.sign(r)*k/(r**2)
            
            ion_fields.append(Ep)

        ion_fields = np.array(ion_fields)
        applied_field = lambda x: 2*U2*x + axial_offset
        applied_field = applied_field(ion_positions)
        
        # stray field + target field + ion field = 0
        stray_field = -1*(ion_fields + applied_field)
        return stray_field
