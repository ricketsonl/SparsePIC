# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:17:42 2016

@author: ricketsonl
"""
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, threadid
cimport openmp
from libc.math cimport sin, cos, sqrt


@cython.boundscheck(False)
@cython.cdivision(True)
def Brotate(double [:,:] vpar, double [:,:] Bpar, double chrg, double dt):
    cdef double a, b, c, d, aa, bb, cc, dd, bc, ad, ac, ab, bd, cd
    cdef double Bmag, sine, theta
    cdef unsigned int i
    cdef unsigned int npar = vpar.shape[0]
    cdef double [:,:] vnew = np.zeros((npar,3))
    
    for i in prange(npar, nogil=True):
        Bmag = sqrt(Bpar[i,0]*Bpar[i,0] + Bpar[i,1]*Bpar[i,1] + Bpar[i,2]*Bpar[i,2])
        theta = Bmag*chrg*dt
        if Bmag > 0:
            sine = sin(theta/2.0)/Bmag
            a = cos(theta/2.0)
            b = Bpar[i,0]*sine; c = Bpar[i,1]*sine; d = Bpar[i,2]*sine
        
            aa = a*a; bb = b*b; cc = c*c; dd = d*d
            bc = b*c; ad = a*d; ac = a*c; ab = a*b; bd = b*d; cd = c*d
        
            vnew[i,0] = (aa+bb-cc-dd)*vpar[i,0] + 2.0*(bc-ad)*vpar[i,1] + 2.0*(bd+ac)*vpar[i,2]
            vnew[i,1] = 2.0*(bc+ad)*vpar[i,0] + (aa+cc-bb-dd)*vpar[i,1] + 2.0*(cd-ab)*vpar[i,2]
            vnew[i,2] = 2.0*(bd-ac)*vpar[i,0] + 2.0*(cd+ab)*vpar[i,1] + (aa+dd-bb-cc)*vpar[i,2]
        else:
            vnew[i,0] = vpar[i,0]; vnew[i,1] = vpar[i,1]; vnew[i,2] = vpar[i,2]
        
    return np.asarray(vnew)