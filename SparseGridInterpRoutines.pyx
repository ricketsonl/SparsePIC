# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:40:43 2016

@author: ricketsonl
"""

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
cimport openmp

###############################################################################
###############################################################################
## Helper functions for bicubic stuff ##
cdef inline double cubic0(double x) nogil:
    return (x-1.0)*(x+1.0)*(x-2.0)/2.0

cdef inline double cubic1(double x) nogil:
    return x*(2.0-x)*(1.0+x)/2.0

cdef inline double cubic2(double x) nogil:
    return x*(x-1.0)*(x+1.0)/6.0

cdef inline double cubicm1(double x) nogil:
    return x*(x-1.0)*(2.0-x)/6.0 
    
cdef inline double QuarticDeltaApprox(double x) nogil:
    return 1.40625 - 4.6875*x*x + 3.28125*x*x*x*x

###############################################################################
###############################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def InterpSparseGrid2DOther(double[:,:] x, double [:] vee, double [:,:] pgrids, double [:,:] mgrids, double [:] dxp, double [:] dyp):
    cdef int npar = x.shape[0]
    cdef int ngrids = pgrids.shape[0]
    cdef int nptsp = pgrids.shape[1]
    cdef int nptsm = mgrids.shape[1]
    
    cdef int nthread = openmp.omp_get_num_procs()
    
    cdef double dxtmp, dytmp
    cdef int ncelx, ncely, ncelym
    cdef int indx, indyp, indym, indxp1, indypp1, indymp1, 
    cdef int indp, indp_xp1, indp_yp1, indp_xp1yp1, indm, indm_xp1, indm_yp1, indm_xp1yp1
    cdef double wtlx, wtrx, wtlyp, wtryp, wtlym, wtrym
    cdef double divp, divm
    cdef double xp, yp, vp
    
    cdef int p, g, i
    
    for g in prange(ngrids-1, nogil=True, num_threads=nthread):
        dxtmp = dxp[g]; dytmp = dyp[g]
        ncelx = int(2**(g+1))
        ncely = int(2**(ngrids-g))
        ncelym = int(ncely/2)
        
        divp = 1./(dxtmp*dytmp*npar)
        divm = 1./(dxtmp*2.*dytmp*npar)
        
        for p in range(npar):
            vp = vee[p]
            wtrx = x[p,0]/dxtmp; wtryp = x[p,1]/dytmp; wtrym = 0.5*wtryp
            indx = <int> wtrx; indyp = <int> wtryp; indym = <int> wtrym
            wtrx = wtrx - indx; wtryp = wtryp - indyp; wtrym = wtrym - indym
            wtlx = 1.0 - wtrx; wtlyp = 1.0 - wtryp; wtlym = 1.0 - wtrym
            
            wtlx = wtlx*vp; wtrx = wtrx*vp
            
            indxp1 = (indx+1) % ncelx; indypp1 = (indyp+1) % ncely; indymp1 = (indym+1) % ncelym
            indp = indx + indyp*ncelx; indp_xp1 = indxp1 + indyp*ncelx; indp_yp1 = indx + indypp1*ncelx; indp_xp1yp1 = indxp1 + indypp1*ncelx
            indm = indx + indym*ncelx; indm_xp1 = indxp1 + indym*ncelx; indm_yp1 = indx + indymp1*ncelx; indm_xp1yp1 = indxp1 + indymp1*ncelx
            
            #print x[p,0], x[p,0] 
            #print wtrx, wtryp, wtlx, wtlyp
            #print indx, indyp, indp, ncelx, ncely
            #print pgrids[g].shape
            
            pgrids[g,indp] += wtlx*wtlyp
            pgrids[g,indp_xp1] += wtrx*wtlyp
            pgrids[g,indp_yp1] += wtlx*wtryp
            pgrids[g,indp_xp1yp1] += wtrx*wtryp
            
            mgrids[g,indm] += wtlx*wtlym
            mgrids[g,indm_xp1] += wtrx*wtlym
            mgrids[g,indm_yp1] += wtlx*wtrym
            mgrids[g,indm_xp1yp1] += wtrx*wtrym
        
        for i in range(nptsp):
            pgrids[g,i] *= divp
        for i in range(nptsm):
            mgrids[g,i] *= divm
            
    g = ngrids-1
    dxtmp = dxp[g]; dytmp = dyp[g]
    ncelx = int(2**(g+1))
    ncely = int(2**(ngrids-g))
    ncelym = int(ncely/2)
    
    divp = 1./(dxtmp*dytmp*npar)
    
    for p in range(npar):
        vp = vee[p]
        wtrx = x[p,0]/dxtmp; wtryp = x[p,1]/dytmp
        indx = <int>wtrx; indyp = <int>wtryp
        wtrx -= indx; wtryp -= indyp
        wtlx = 1.0 - wtrx; wtlyp = 1.0 - wtryp
        
        indxp1 = (indx+1) % ncelx; indypp1 = (indyp+1) % ncely
        indp = indx + indyp*ncelx; indp_xp1 = indxp1 + indyp*ncelx; indp_yp1 = indx + indypp1*ncelx; indp_xp1yp1 = indxp1 + indypp1*ncelx
        
        wtlx *= vp; wtrx *= vp
        
        pgrids[g,indp] += wtlx*wtlyp
        pgrids[g,indp_xp1] += wtrx*wtlyp
        pgrids[g,indp_yp1] += wtlx*wtryp
        pgrids[g,indp_xp1yp1] += wtrx*wtryp
    
    for i in prange(nptsp, nogil=True, num_threads=nthread):
        pgrids[g,i] *= divp

###############################################################################
###############################################################################
@cython.boundscheck(False)
@cython.wraparound(False)
def InterpSparseGrid2Dbicubic(double[:,:] x, double [:] vee, double [:,:] pgrids, double [:,:] mgrids, double [:] dxp, double [:] dyp):
    cdef int npar = x.shape[0]
    cdef int ngrids = pgrids.shape[0]
    cdef int nptsp = pgrids.shape[1]
    cdef int nptsm = mgrids.shape[1]
    
    cdef int nthread = openmp.omp_get_num_procs()
    
    cdef double dxtmp, dytmp
    cdef int ncelx, ncely, ncelym
    cdef int indx, indyp, indym, indxp1, indypp1, indymp1, 
    cdef int indp, indp_xp1, indp_yp1, indp_xp1yp1, indm, indm_xp1, indm_yp1, indm_xp1yp1
    cdef double wtlx, wtrx, wtlyp, wtryp, wtlym, wtrym
    cdef double divp, divm
    cdef double xp, yp, vp
    
    cdef int p, g, i
    
    for g in prange(ngrids-1, nogil=True, num_threads=nthread):
        dxtmp = dxp[g]; dytmp = dyp[g]
        ncelx = int(2**(g+1))
        ncely = int(2**(ngrids-g))
        ncelym = int(ncely/2)
        
        divp = 1./(dxtmp*dytmp*npar)
        divm = 1./(dxtmp*2.*dytmp*npar)
        
        for p in range(npar):
            vp = vee[p]
            wtrx = x[p,0]/dxtmp; wtryp = x[p,1]/dytmp; wtrym = 0.5*wtryp
            indx = <int> wtrx; indyp = <int> wtryp; indym = <int> wtrym
            wtrx = wtrx - indx; wtryp = wtryp - indyp; wtrym = wtrym - indym
            wtlx = 1.0 - wtrx; wtlyp = 1.0 - wtryp; wtlym = 1.0 - wtrym
            
            wtlx = QuarticDeltaApprox(1.0-wtlx); wtrx = QuarticDeltaApprox(1.0-wtrx)
            wtlyp = QuarticDeltaApprox(1.0-wtlyp); wtryp = QuarticDeltaApprox(1.0-wtryp)
            wtlym = QuarticDeltaApprox(1.0-wtlym); wtrym = QuarticDeltaApprox(1.0-wtrym)
            
            wtlx = wtlx*vp; wtrx = wtrx*vp
            
            indxp1 = (indx+1) % ncelx; indypp1 = (indyp+1) % ncely; indymp1 = (indym+1) % ncelym
            indp = indx + indyp*ncelx; indp_xp1 = indxp1 + indyp*ncelx; indp_yp1 = indx + indypp1*ncelx; indp_xp1yp1 = indxp1 + indypp1*ncelx
            indm = indx + indym*ncelx; indm_xp1 = indxp1 + indym*ncelx; indm_yp1 = indx + indymp1*ncelx; indm_xp1yp1 = indxp1 + indymp1*ncelx
            
            #print x[p,0], x[p,0] 
            #print wtrx, wtryp, wtlx, wtlyp
            #print indx, indyp, indp, ncelx, ncely
            #print pgrids[g].shape
            
            pgrids[g,indp] += wtlx*wtlyp
            pgrids[g,indp_xp1] += wtrx*wtlyp
            pgrids[g,indp_yp1] += wtlx*wtryp
            pgrids[g,indp_xp1yp1] += wtrx*wtryp
            
            mgrids[g,indm] += wtlx*wtlym
            mgrids[g,indm_xp1] += wtrx*wtlym
            mgrids[g,indm_yp1] += wtlx*wtrym
            mgrids[g,indm_xp1yp1] += wtrx*wtrym
        
        for i in range(nptsp):
            pgrids[g,i] *= divp
        for i in range(nptsm):
            mgrids[g,i] *= divm
            
    g = ngrids-1
    dxtmp = dxp[g]; dytmp = dyp[g]
    ncelx = int(2**(g+1))
    ncely = int(2**(ngrids-g))
    ncelym = int(ncely/2)
    
    divp = 1./(dxtmp*dytmp*npar)
    
    for p in range(npar):
        vp = vee[p]
        wtrx = x[p,0]/dxtmp; wtryp = x[p,1]/dytmp
        indx = <int>wtrx; indyp = <int>wtryp
        wtrx -= indx; wtryp -= indyp
        wtlx = 1.0 - wtrx; wtlyp = 1.0 - wtryp
        
        wtlx = QuarticDeltaApprox(1.0-wtlx); wtrx = QuarticDeltaApprox(1.0-wtrx)
        wtlyp = QuarticDeltaApprox(1.0-wtlyp); wtryp = QuarticDeltaApprox(1.0-wtryp)
        
        indxp1 = (indx+1) % ncelx; indypp1 = (indyp+1) % ncely
        indp = indx + indyp*ncelx; indp_xp1 = indxp1 + indyp*ncelx; indp_yp1 = indx + indypp1*ncelx; indp_xp1yp1 = indxp1 + indypp1*ncelx
        
        wtlx *= vp; wtrx *= vp
        
        pgrids[g,indp] += wtlx*wtlyp
        pgrids[g,indp_xp1] += wtrx*wtlyp
        pgrids[g,indp_yp1] += wtlx*wtryp
        pgrids[g,indp_xp1yp1] += wtrx*wtryp
    
    for i in prange(nptsp, nogil=True, num_threads=nthread):
        pgrids[g,i] *= divp

###############################################################################
###############################################################################
      
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void InterpParSinglePar3D(double xp, double yp, double zp, double[:,:] grids, double[:] parvec, int p, int g, int ncx, int ncy, int ncz, double[:] Ls, double mult) nogil:
    cdef double wtlx, wtly, wtlz, wtrx, wtry, wtrz
    cdef int indx, indy, indz, indxp1, indyp1, indzp1
    cdef double dx, dy, dz
    dx = Ls[0]/ncx; dy = Ls[1]/ncy; dz = Ls[2]/ncz
    wtrx = xp/dx; wtry = yp/dy; wtrz = zp/dz
    indx = <int> wtrx; indy = <int> wtry; indz = <int> wtrz
    wtrx = wtrx - indx; wtry = wtry - indy; wtrz = wtrz - indz
    wtlx = 1.0 - wtrx; wtly = 1.0 - wtry; wtlz = 1.0 - wtrz
    
    indxp1 = (indx+1) % ncx; indyp1 = (indy+1) % ncy; indzp1 = (indz+1) % ncz
    
    cdef int ind, ind_xp1, ind_yp1, ind_zp1, ind_xyp1, ind_xzp1, ind_yzp1, ind_xyzp1
    
    ind = indx + (indy + indz*ncy)*ncx
    ind_xp1 = indxp1 + (indy + indz*ncy)*ncx; ind_yp1 = indx + (indyp1 + indz*ncy)*ncx; ind_zp1 = indx + (indy + indzp1*ncy)*ncx
    ind_xyp1 = indxp1 + (indyp1 + indz*ncy)*ncx; ind_xzp1 = indxp1 + (indy + indzp1*ncy)*ncx; ind_yzp1 = indx + (indyp1 + indzp1*ncy)*ncx
    ind_xyzp1 = indxp1 + (indyp1 + indzp1*ncy)*ncx
    
    parvec[p] += mult*wtlx*wtly*(wtlz*grids[g,ind] + wtrz*grids[g,ind_zp1])
    parvec[p] += mult*wtrx*wtly*(wtlz*grids[g,ind_xp1] + wtrz*grids[g,ind_xzp1])
    parvec[p] += mult*wtlx*wtry*(wtlz*grids[g,ind_yp1] + wtrz*grids[g,ind_yzp1])
    parvec[p] += mult*wtrx*wtry*(wtlz*grids[g,ind_xyp1] + wtrz*grids[g,ind_xyzp1])
    
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline void InterpParSinglePar3D_RG(double xp, double yp, double zp, double[:,:] grids, double[:,:,:] gvec, int i, int j, int k, int g, int ncx, int ncy, int ncz, double[:] Ls, double mult) nogil:
    cdef double wtrx, wtry, wtrz, wtlx, wtly, wtlz
    cdef int indx, indy, indz, indxp1, indyp1, indzp1
    cdef double dx, dy, dz
    dx = Ls[0]/ncx; dy = Ls[1]/ncy; dz = Ls[2]/ncz
    wtrx = xp/dx; wtry = yp/dy; wtrz = zp/dz
    indx = <int> wtrx; indy = <int> wtry; indz = <int> wtrz
    wtrx = wtrx - indx; wtry = wtry - indy; wtrz = wtrz - indz
    wtlx = 1.0 - wtrx; wtly = 1.0 - wtry; wtlz = 1.0 - wtrz
    
    indxp1 = (indx+1) % ncx; indyp1 = (indy+1) % ncy; indzp1 = (indz+1) % ncz
    
    cdef int ind, ind_xp1, ind_yp1, ind_zp1, ind_xyp1, ind_xzp1, ind_yzp1, ind_xyzp1
    
    ind = indx + (indy + indz*ncy)*ncx
    ind_xp1 = indxp1 + (indy + indz*ncy)*ncx; ind_yp1 = indx + (indyp1 + indz*ncy)*ncx; ind_zp1 = indx + (indy + indzp1*ncy)*ncx
    ind_xyp1 = indxp1 + (indyp1 + indz*ncy)*ncx; ind_xzp1 = indxp1 + (indy + indzp1*ncy)*ncx; ind_yzp1 = indx + (indyp1 + indzp1*ncy)*ncx
    ind_xyzp1 = indxp1 + (indyp1 + indzp1*ncy)*ncx
    
    gvec[i,j,k] += mult*wtlx*wtly*(wtlz*grids[g,ind] + wtrz*grids[g,ind_zp1])
    gvec[i,j,k] += mult*wtrx*wtly*(wtlz*grids[g,ind_xp1] + wtrz*grids[g,ind_xzp1])
    gvec[i,j,k] += mult*wtlx*wtry*(wtlz*grids[g,ind_yp1] + wtrz*grids[g,ind_yzp1])
    gvec[i,j,k] += mult*wtrx*wtry*(wtlz*grids[g,ind_xyp1] + wtrz*grids[g,ind_xyzp1])

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline double InterpParSinglePar3Dbicubic_RG(double xp, double yp, double zp, double[:,:] grids, int g, int ncx, int ncy, int ncz, double[:] Ls, double mult) nogil:
    cdef double wtrx, wtry, wtrz
    cdef int indx, indy, indz, indxp1, indyp1, indzp1, indxm1, indym1, indzm1, indxp2, indyp2, indzp2
    
    cdef int i, i_zp1, i_zp2, i_zm1, i_yp1, i_yp1zp1, i_yp1zm1, i_yp1zp2, i_yp2, i_yp2zp1, i_yp2zp2, i_yp2zm1, i_ym1, i_ym1zp1, i_ym1zp2, i_ym1zm1
    cdef int i_xp1, i_xp1zp1, i_xp1zp2, i_xp1zm1, i_xp1yp1, i_xp1yp1zp1, i_xp1yp1zm1, i_xp1yp1zp2, i_xp1yp2, i_xp1yp2zp1, i_xp1yp2zp2, i_xp1yp2zm1, i_xp1ym1, i_xp1ym1zp1, i_xp1ym1zp2, i_xp1ym1zm1
    cdef int i_xp2, i_xp2zp1, i_xp2zp2, i_xp2zm1, i_xp2yp1, i_xp2yp1zp1, i_xp2yp1zm1, i_xp2yp1zp2, i_xp2yp2, i_xp2yp2zp1, i_xp2yp2zp2, i_xp2yp2zm1, i_xp2ym1, i_xp2ym1zp1, i_xp2ym1zp2, i_xp2ym1zm1
    cdef int i_xm1, i_xm1zp1, i_xm1zp2, i_xm1zm1, i_xm1yp1, i_xm1yp1zp1, i_xm1yp1zm1, i_xm1yp1zp2, i_xm1yp2, i_xm1yp2zp1, i_xm1yp2zp2, i_xm1yp2zm1, i_xm1ym1, i_xm1ym1zp1, i_xm1ym1zp2, i_xm1ym1zm1
    
    cdef double wx0, wxp1, wxp2, wxm1, wy0, wyp1, wyp2, wym1, wz0, wzp1, wzp2, wzm1    
    
    cdef double dx, dy, dz
    cdef double gridval = 0.0
    dx = Ls[0]/ncx; dy = Ls[1]/ncy; dz = Ls[2]/ncz
    wtrx = xp/dx; wtry = yp/dy; wtrz = zp/dz
    indx = <int> wtrx; indy = <int> wtry; indz = <int> wtrz
    wtrx = wtrx - indx; wtry = wtry - indy; wtrz = wtrz - indz
    
    wx0 = cubic0(wtrx); wxp1 = cubic1(wtrx); wxp2 = cubic2(wtrx); wxm1 = cubicm1(wtrx)
    wy0 = cubic0(wtry); wyp1 = cubic1(wtry); wyp2 = cubic2(wtry); wym1 = cubicm1(wtry)
    wz0 = cubic0(wtrz); wzp1 = cubic1(wtrz); wzp2 = cubic2(wtrz); wzm1 = cubicm1(wtrz)
    
    indxp1 = (indx+1)%ncx; indxp2 = (indx+2)%ncx; indxm1 = (indx-1)%ncx
    indyp1 = (indy+1)%ncy; indyp2 = (indy+2)%ncy; indym1 = (indy-1)%ncy
    indzp1 = (indz+1)%ncz; indzp2 = (indz+2)%ncz; indzm1 = (indz-1)%ncz
    
    i = indx + (indy + indz*ncy)*ncx; i_zp1 = indx + (indy + indzp1*ncy)*ncx
    i_zp2 = indx + (indy + indzp2*ncy)*ncx; i_zm1 = indx + (indy + indzm1*ncy)*ncx
    i_yp1 = indx + (indyp1 + indz*ncy)*ncx; i_yp1zp1 = indx + (indyp1 + indzp1*ncy)*ncx
    i_yp1zp2 = indx + (indyp1 + indzp2*ncy)*ncx; i_yp1zm1 = indx + (indyp1 + indzm1*ncy)*ncx
    i_yp2 = indx + (indyp2 + indz*ncy)*ncx; i_yp2zp1 = indx + (indyp2 + indzp1*ncy)*ncx
    i_yp2zp2 = indx + (indyp2 + indzp2*ncy)*ncx; i_yp2zm1 = indx + (indyp2 + indzm1*ncy)*ncx
    i_ym1 = indx + (indym1 + indz*ncy)*ncx; i_ym1zp1 = indx + (indym1 + indzp1*ncy)*ncx
    i_ym1zp2 = indx + (indym1 + indzp2*ncy)*ncx; i_ym1zm1 = indx + (indym1 + indzm1*ncy)*ncx
    
    i_xp1 = indxp1 + (indy + indz*ncy)*ncx; i_xp1zp1 = indxp1 + (indy + indzp1*ncy)*ncx
    i_xp1zp2 = indxp1 + (indy + indzp2*ncy)*ncx; i_xp1zm1 = indxp1 + (indy + indzm1*ncy)*ncx
    i_xp1yp1 = indxp1 + (indyp1 + indz*ncy)*ncx; i_xp1yp1zp1 = indxp1 + (indyp1 + indzp1*ncy)*ncx
    i_xp1yp1zp2 = indxp1 + (indyp1 + indzp2*ncy)*ncx; i_xp1yp1zm1 = indxp1 + (indyp1 + indzm1*ncy)*ncx
    i_xp1yp2 = indxp1 + (indyp2 + indz*ncy)*ncx; i_xp1yp2zp1 = indxp1 + (indyp2 + indzp1*ncy)*ncx
    i_xp1yp2zp2 = indxp1 + (indyp2 + indzp2*ncy)*ncx; i_xp1yp2zm1 = indxp1 + (indyp2 + indzm1*ncy)*ncx
    i_xp1ym1 = indxp1 + (indym1 + indz*ncy)*ncx; i_xp1ym1zp1 = indxp1 + (indym1 + indzp1*ncy)*ncx
    i_xp1ym1zp2 = indxp1 + (indym1 + indzp2*ncy)*ncx; i_xp1ym1zm1 = indxp1 + (indym1 + indzm1*ncy)*ncx
    
    i_xp2 = indxp2 + (indy + indz*ncy)*ncx; i_xp2zp1 = indxp2 + (indy + indzp1*ncy)*ncx
    i_xp2zp2 = indxp2 + (indy + indzp2*ncy)*ncx; i_xp2zm1 = indxp2 + (indy + indzm1*ncy)*ncx
    i_xp2yp1 = indxp2 + (indyp1 + indz*ncy)*ncx; i_xp2yp1zp1 = indxp2 + (indyp1 + indzp1*ncy)*ncx
    i_xp2yp1zp2 = indxp2 + (indyp1 + indzp2*ncy)*ncx; i_xp2yp1zm1 = indxp2 + (indyp1 + indzm1*ncy)*ncx
    i_xp2yp2 = indxp2 + (indyp2 + indz*ncy)*ncx; i_xp2yp2zp1 = indxp2 + (indyp2 + indzp1*ncy)*ncx
    i_xp2yp2zp2 = indxp2 + (indyp2 + indzp2*ncy)*ncx; i_xp2yp2zm1 = indxp2 + (indyp2 + indzm1*ncy)*ncx
    i_xp2ym1 = indxp2 + (indym1 + indz*ncy)*ncx; i_xp2ym1zp1 = indxp2 + (indym1 + indzp1*ncy)*ncx
    i_xp2ym1zp2 = indxp2 + (indym1 + indzp2*ncy)*ncx; i_xp2ym1zm1 = indxp2 + (indym1 + indzm1*ncy)*ncx
    
    i_xm1 = indxm1 + (indy + indz*ncy)*ncx; i_xm1zp1 = indxm1 + (indy + indzp1*ncy)*ncx
    i_xm1zp2 = indxm1 + (indy + indzp2*ncy)*ncx; i_xm1zm1 = indxm1 + (indy + indzm1*ncy)*ncx
    i_xm1yp1 = indxm1 + (indyp1 + indz*ncy)*ncx; i_xm1yp1zp1 = indxm1 + (indyp1 + indzp1*ncy)*ncx
    i_xm1yp1zp2 = indxm1 + (indyp1 + indzp2*ncy)*ncx; i_xm1yp1zm1 = indxm1 + (indyp1 + indzm1*ncy)*ncx
    i_xm1yp2 = indxm1 + (indyp2 + indz*ncy)*ncx; i_xm1yp2zp1 = indxm1 + (indyp2 + indzp1*ncy)*ncx
    i_xm1yp2zp2 = indxm1 + (indyp2 + indzp2*ncy)*ncx; i_xm1yp2zm1 = indxm1 + (indyp2 + indzm1*ncy)*ncx
    i_xm1ym1 = indxm1 + (indym1 + indz*ncy)*ncx; i_xm1ym1zp1 = indxm1 + (indym1 + indzp1*ncy)*ncx
    i_xm1ym1zp2 = indxm1 + (indym1 + indzp2*ncy)*ncx; i_xm1ym1zm1 = indxm1 + (indym1 + indzm1*ncy)*ncx    
    
    gridval += wx0*wy0*(wz0*grids[g,i] + wzp1*grids[g,i_zp1] + wzp2*grids[g,i_zp2] + wzm1*grids[g,i_zm1])
    gridval += wx0*wyp1*(wz0*grids[g,i_yp1] + wzp1*grids[g,i_yp1zp1] + wzp2*grids[g,i_yp1zp2] + wzm1*grids[g,i_yp1zm1])
    gridval += wx0*wyp2*(wz0*grids[g,i_yp2] + wzp1*grids[g,i_yp2zp1] + wzp2*grids[g,i_yp2zp2] + wzm1*grids[g,i_yp2zm1])
    gridval += wx0*wym1*(wz0*grids[g,i_ym1] + wzp1*grids[g,i_ym1zp1] + wzp2*grids[g,i_ym1zp2] + wzm1*grids[g,i_ym1zm1])
    
    gridval += wxp1*wy0*(wz0*grids[g,i_xp1] + wzp1*grids[g,i_xp1zp1] + wzp2*grids[g,i_xp1zp2] + wzm1*grids[g,i_xp1zm1])
    gridval += wxp1*wyp1*(wz0*grids[g,i_xp1yp1] + wzp1*grids[g,i_xp1yp1zp1] + wzp2*grids[g,i_xp1yp1zp2] + wzm1*grids[g,i_xp1yp1zm1])
    gridval += wxp1*wyp2*(wz0*grids[g,i_xp1yp2] + wzp1*grids[g,i_xp1yp2zp1] + wzp2*grids[g,i_xp1yp2zp2] + wzm1*grids[g,i_xp1yp2zm1])
    gridval += wxp1*wym1*(wz0*grids[g,i_xp1ym1] + wzp1*grids[g,i_xp1ym1zp1] + wzp2*grids[g,i_xp1ym1zp2] + wzm1*grids[g,i_xp1ym1zm1])    
    
    gridval += wxp2*wy0*(wz0*grids[g,i_xp2] + wzp1*grids[g,i_xp2zp1] + wzp2*grids[g,i_xp2zp2] + wzm1*grids[g,i_xp2zm1])
    gridval += wxp2*wyp1*(wz0*grids[g,i_xp2yp1] + wzp1*grids[g,i_xp2yp1zp1] + wzp2*grids[g,i_xp2yp1zp2] + wzm1*grids[g,i_xp2yp1zm1])
    gridval += wxp2*wyp2*(wz0*grids[g,i_xp2yp2] + wzp1*grids[g,i_xp2yp2zp1] + wzp2*grids[g,i_xp2yp2zp2] + wzm1*grids[g,i_xp2yp2zm1])
    gridval += wxp2*wym1*(wz0*grids[g,i_xp2ym1] + wzp1*grids[g,i_xp2ym1zp1] + wzp2*grids[g,i_xp2ym1zp2] + wzm1*grids[g,i_xp2ym1zm1])    
    
    gridval += wxm1*wy0*(wz0*grids[g,i_xm1] + wzp1*grids[g,i_xm1zp1] + wzp2*grids[g,i_xm1zp2] + wzm1*grids[g,i_xm1zm1])
    gridval += wxm1*wyp1*(wz0*grids[g,i_xm1yp1] + wzp1*grids[g,i_xm1yp1zp1] + wzp2*grids[g,i_xm1yp1zp2] + wzm1*grids[g,i_xm1yp1zm1])
    gridval += wxm1*wyp2*(wz0*grids[g,i_xm1yp2] + wzp1*grids[g,i_xm1yp2zp1] + wzp2*grids[g,i_xm1yp2zp2] + wzm1*grids[g,i_xm1yp2zm1])
    gridval += wxm1*wym1*(wz0*grids[g,i_xm1ym1] + wzp1*grids[g,i_xm1ym1zp1] + wzp2*grids[g,i_xm1ym1zp2] + wzm1*grids[g,i_xm1ym1zm1])    
    
    gridval *= mult    
    
    return gridval
###############################################################################
###############################################################################

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def EvaluateAt3D(double[:,:] x, double[:,:] Fnp2grids, double[:,:] Fnp1grids, double[:,:] Fngrids, int[:,:] ncnp2, int[:,:] ncnp1, int[:,:] ncn, double[:] Ls):
    cdef int npar = x.shape[0]
    
    #cdef double Lx = Ls[0]
    #cdef double Ly = Ls[1]
    #cdef double Lz = Ls[2]
    
    cdef int ngrids_np2 = Fnp2grids.shape[0]
    cdef int ngrids_np1 = Fnp1grids.shape[0]
    cdef int ngrids_n = Fngrids.shape[0]
    
    #cdef int npts_np2 = Fnp2grids.shape[1]
    #cdef int npts_np1 = Fnp1grids.shape[1]
    #cdef int npts_n = Fngrids.shape[1]
    
    cdef int p, g
    
    cdef int nthread = openmp.omp_get_num_procs()
    cdef double xp, yp, zp
    
    cdef double[:] Fpar = np.zeros(npar)
    
    for p in prange(npar, nogil=True, num_threads=nthread):
        xp = x[p,0]; yp = x[p,1]; zp = x[p,2]
        
        for g in range(ngrids_n):
            InterpParSinglePar3D(xp,yp,zp,Fnp2grids,Fpar,p,g,ncnp2[0,g],ncnp2[1,g],ncnp2[2,g],Ls,1.0)
            InterpParSinglePar3D(xp,yp,zp,Fnp1grids,Fpar,p,g,ncnp1[0,g],ncnp1[1,g],ncnp1[2,g],Ls,-2.0)
            InterpParSinglePar3D(xp,yp,zp,Fngrids,Fpar,p,g,ncn[0,g],ncn[1,g],ncn[2,g],Ls,1.0)
            
        for g in range(ngrids_n,ngrids_np1):
            InterpParSinglePar3D(xp,yp,zp,Fnp2grids,Fpar,p,g,ncnp2[0,g],ncnp2[1,g],ncnp2[2,g],Ls,1.0)
            InterpParSinglePar3D(xp,yp,zp,Fnp1grids,Fpar,p,g,ncnp1[0,g],ncnp1[1,g],ncnp1[2,g],Ls,-2.0)
        
        for g in range(ngrids_np1,ngrids_np2):
            InterpParSinglePar3D(xp,yp,zp,Fnp2grids,Fpar,p,g,ncnp2[0,g],ncnp2[1,g],ncnp2[2,g],Ls,1.0)
            
    return np.asarray(Fpar)

###############################################################################
###############################################################################
   
@cython.wraparound(False)
@cython.boundscheck(False)
def EvaluateOnGrid3Dbicubic(double[:] x, double[:] y, double[:] z, double[:,:] np2grids, double[:,:] np1grids, double[:,:] ngrids, int [:,:] ncnp2, int [:,:] ncnp1, int [:,:] ncn, double [:] Ls):
    cdef int nx = x.shape[0]
    cdef int ny = y.shape[0]
    cdef int nz = z.shape[0]
    
    cdef int ngrids_np2 = np2grids.shape[0]
    cdef int ngrids_np1 = np1grids.shape[0]
    cdef int ngrids_n = ngrids.shape[0]
    
    cdef double[:,:,:] gvals = np.zeros((nx,ny,nz))
    cdef double xp, yp, zp
    cdef int i, j, k, g
    cdef int nthread = openmp.omp_get_num_procs()
    
    for i in prange(nx, nogil=True):
        xp = x[i]
        for j in range(ny):
            yp = y[j]
            for k in range(nz):
                zp = z[k]
                
                for g in range(ngrids_n):
                    gvals[i,j,k] += InterpParSinglePar3Dbicubic_RG(xp,yp,zp,np2grids,g,ncnp2[0,g],ncnp2[1,g],ncnp2[2,g],Ls,1.0)
                    gvals[i,j,k] += InterpParSinglePar3Dbicubic_RG(xp,yp,zp,np1grids,g,ncnp1[0,g],ncnp1[1,g],ncnp1[2,g],Ls,-2.0)
                    gvals[i,j,k] += InterpParSinglePar3Dbicubic_RG(xp,yp,zp,ngrids,g,ncn[0,g],ncn[1,g],ncn[2,g],Ls,1.0)
            
                for g in range(ngrids_n,ngrids_np1):
                    gvals[i,j,k] += InterpParSinglePar3Dbicubic_RG(xp,yp,zp,np2grids,g,ncnp2[0,g],ncnp2[1,g],ncnp2[2,g],Ls,1.0)
                    gvals[i,j,k] += InterpParSinglePar3Dbicubic_RG(xp,yp,zp,np1grids,g,ncnp1[0,g],ncnp1[1,g],ncnp1[2,g],Ls,-2.0)
        
                for g in range(ngrids_np1,ngrids_np2):
                    gvals[i,j,k] += InterpParSinglePar3Dbicubic_RG(xp,yp,zp,np2grids,g,ncnp2[0,g],ncnp2[1,g],ncnp2[2,g],Ls,1.0)
                
    return np.asarray(gvals)
    
###############################################################################
###############################################################################
   
@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def EvaluateOnGrid3D(double[:] x, double[:] y, double[:] z, double[:,:] np2grids, double[:,:] np1grids, double[:,:] ngrids, int [:,:] ncnp2, int [:,:] ncnp1, int [:,:] ncn, double [:] Ls):
    cdef int nx = x.shape[0]
    cdef int ny = y.shape[0]
    cdef int nz = z.shape[0]
    
    cdef int ngrids_np2 = np2grids.shape[0]
    cdef int ngrids_np1 = np1grids.shape[0]
    cdef int ngrids_n = ngrids.shape[0]
    
    cdef double[:,:,:] gvals = np.zeros((nx,ny,nz))
    cdef double xp, yp, zp
    cdef int i, j, k, g
    cdef int nthread = openmp.omp_get_num_procs()
    
    for i in prange(nx, nogil=True):
        xp = x[i]
        for j in range(ny):
            yp = y[j]
            for k in range(nz):
                zp = z[k]
                
                for g in range(ngrids_n):
                    InterpParSinglePar3D_RG(xp,yp,zp,np2grids,gvals,i,j,k,g,ncnp2[0,g],ncnp2[1,g],ncnp2[2,g],Ls,1.0)
                    InterpParSinglePar3D_RG(xp,yp,zp,np1grids,gvals,i,j,k,g,ncnp1[0,g],ncnp1[1,g],ncnp1[2,g],Ls,-2.0)
                    InterpParSinglePar3D_RG(xp,yp,zp,ngrids,gvals,i,j,k,g,ncn[0,g],ncn[1,g],ncn[2,g],Ls,1.0)
            
                for g in range(ngrids_n,ngrids_np1):
                    InterpParSinglePar3D_RG(xp,yp,zp,np2grids,gvals,i,j,k,g,ncnp2[0,g],ncnp2[1,g],ncnp2[2,g],Ls,1.0)
                    InterpParSinglePar3D_RG(xp,yp,zp,np1grids,gvals,i,j,k,g,ncnp1[0,g],ncnp1[1,g],ncnp1[2,g],Ls,-2.0)
        
                for g in range(ngrids_np1,ngrids_np2):
                    InterpParSinglePar3D_RG(xp,yp,zp,np2grids,gvals,i,j,k,g,ncnp2[0,g],ncnp2[1,g],ncnp2[2,g],Ls,1.0)
                
    return np.asarray(gvals)
    
###############################################################################
###############################################################################
    
@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cdef inline void InterpGridSinglePar3D(double xp, double yp, double zp, double vp, double[:,:] grids, int g, int ncx, int ncy, int ncz, double[:] Ls) nogil:
    cdef double wtlx, wtly, wtlz, wtrx, wtry, wtrz
    cdef int indx, indy, indz, indxp1, indyp1, indzp1
    cdef double dx, dy, dz
    dx = Ls[0]/ncx; dy = Ls[1]/ncy; dz = Ls[2]/ncz
    wtrx = xp/dx; wtry = yp/dy; wtrz = zp/dz
    indx = <int> wtrx; indy = <int> wtry; indz = <int> wtrz
    wtrx = wtrx - indx; wtry = wtry - indy; wtrz = wtrz - indz
    wtlx = 1.0 - wtrx; wtly = 1.0 - wtry; wtlz = 1.0 - wtrz
    
    wtlx = wtlx*vp; wtrx = wtrx*vp
    
    indxp1 = (indx+1) % ncx; indyp1 = (indy+1) % ncy; indzp1 = (indz+1) % ncz
    
    cdef int ind, ind_xp1, ind_yp1, ind_zp1, ind_xyp1, ind_xzp1, ind_yzp1, ind_xyzp1
    
    ind = indx + (indy + indz*ncy)*ncx
    ind_xp1 = indxp1 + (indy + indz*ncy)*ncx; ind_yp1 = indx + (indyp1 + indz*ncy)*ncx; ind_zp1 = indx + (indy + indzp1*ncy)*ncx
    ind_xyp1 = indxp1 + (indyp1 + indz*ncy)*ncx; ind_xzp1 = indxp1 + (indy + indzp1*ncy)*ncx; ind_yzp1 = indx + (indyp1 + indzp1*ncy)*ncx
    ind_xyzp1 = indxp1 + (indyp1 + indzp1*ncy)*ncx
    
    grids[g,ind] += wtlx*wtly*wtlz
    grids[g,ind_xp1] += wtrx*wtly*wtlz
    grids[g,ind_yp1] += wtlx*wtry*wtlz
    grids[g,ind_zp1] += wtlx*wtly*wtrz
    grids[g,ind_xyp1] += wtrx*wtry*wtlz
    grids[g,ind_xzp1] += wtrx*wtly*wtrz
    grids[g,ind_yzp1] += wtlx*wtry*wtrz
    grids[g,ind_xyzp1] += wtrx*wtry*wtrz

###############################################################################
###############################################################################
@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
def InterpSparseGrid3D(double[:,:] x, double[:] v, double [:,:] np2grids, double [:,:] np1grids, double [:,:] ngrids, int [:,:] ncnp2, int [:,:] ncnp1, int [:,:] ncn, double [:] Ls):
    cdef int npar = x.shape[0]
    
    cdef int ngrids_np2 = np2grids.shape[0]
    cdef int ngrids_np1 = np1grids.shape[0]
    cdef int ngrids_n = ngrids.shape[0]
    
    cdef int npts_np2 = np2grids.shape[1]
    cdef int npts_np1 = np1grids.shape[1]
    cdef int npts_n = ngrids.shape[1]
    
    cdef int nthread = openmp.omp_get_num_procs()
    cdef double vp, xp, yp, zp
    
    cdef double Lx = Ls[0], Ly = Ls[1], Lz = Ls[2]
    
    cdef int p, g, i
    
    cdef double div_np2, div_np1, div_n, dx_np2, dy_np2, dz_np2, dx_np1, dy_np1, dz_np1, dx_n, dy_n, dz_n
    
    for g in prange(ngrids_n, nogil=True, num_threads=nthread):
        dx_np2 = Lx/ncnp2[0,g]; dy_np2 = Ly/ncnp2[1,g]; dz_np2 = Lz/ncnp2[2,g]
        dx_np1 = Lx/ncnp1[0,g]; dy_np1 = Ly/ncnp1[1,g]; dz_np1 = Lz/ncnp1[2,g]
        dx_n = Lx/ncn[0,g]; dy_n = Ly/ncn[1,g]; dz_n = Lz/ncn[2,g]
        
        div_np2 = 1./(dx_np2*dy_np2*dz_np2*npar)
        div_np1 = 1./(dx_np1*dy_np1*dz_np1*npar)
        div_n = 1./(dx_n*dy_n*dz_n*npar)
        
        for p in range(npar):
            vp = v[p]; xp = x[p,0]; yp = x[p,1]; zp = x[p,2]
            
            InterpGridSinglePar3D(xp,yp,zp,vp,np2grids,g,ncnp2[0,g],ncnp2[1,g],ncnp2[2,g],Ls)
            InterpGridSinglePar3D(xp,yp,zp,vp,np1grids,g,ncnp1[0,g],ncnp1[1,g],ncnp1[2,g],Ls)
            InterpGridSinglePar3D(xp,yp,zp,vp,ngrids,g,ncn[0,g],ncn[1,g],ncn[2,g],Ls)
            
        for i in range(npts_np2):
            np2grids[g,i] *= div_np2
        for i in range(npts_np1):
            np1grids[g,i] *= div_np1
        for i in range(npts_n):
            ngrids[g,i] *= div_n
        
    for g in prange(ngrids_n,ngrids_np1, nogil=True, num_threads=nthread):
        dx_np2 = Lx/ncnp2[0,g]; dy_np2 = Ly/ncnp2[1,g]; dz_np2 = Lz/ncnp2[2,g]
        dx_np1 = Lx/ncnp1[0,g]; dy_np1 = Ly/ncnp1[1,g]; dz_np1 = Lz/ncnp1[2,g]
        
        div_np2 = 1./(dx_np2*dy_np2*dz_np2*npar)
        div_np1 = 1./(dx_np1*dy_np1*dz_np1*npar)
        
        for p in range(npar):
            vp = v[p]; xp = x[p,0]; yp = x[p,1]; zp = x[p,2]
            
            InterpGridSinglePar3D(xp,yp,zp,vp,np2grids,g,ncnp2[0,g],ncnp2[1,g],ncnp2[2,g],Ls)
            InterpGridSinglePar3D(xp,yp,zp,vp,np1grids,g,ncnp1[0,g],ncnp1[1,g],ncnp1[2,g],Ls)
            
        for i in range(npts_np2):
            np2grids[g,i] *= div_np2
        for i in range(npts_np1):
            np1grids[g,i] *= div_np1
            
    for g in prange(ngrids_np1,ngrids_np2, nogil=True, num_threads=nthread):
        dx_np2 = Lx/ncnp2[0,g]; dy_np2 = Ly/ncnp2[1,g]; dz_np2 = Lz/ncnp2[2,g]
        
        div_np2 = 1./(dx_np2*dy_np2*dz_np2*npar)
        
        for p in range(npar):
            vp = v[p]; xp = x[p,0]; yp = x[p,1]; zp = x[p,2]
            InterpGridSinglePar3D(xp,yp,zp,vp,np2grids,g,ncnp2[0,g],ncnp2[1,g],ncnp2[2,g],Ls)
            
        for i in range(npts_np2):
            np2grids[g,i] *= div_np2

###############################################################################
###############################################################################
            
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def EvaluateOnGrid2D(double[:] x, double[:] y, double[:,:] pgrids, double[:,:] mgrids, double[:] dxp, double [:] dyp):
    cdef int ngx = x.shape[0]
    cdef int ngy = y.shape[0]
    cdef int ngrids = pgrids.shape[0]
    cdef int ncelmax = mgrids.shape[1]
    
    cdef int ncelx, ncely, ncelym
    cdef double dxtmp, dytmpp, dytmpm
    
    cdef int indx, indyp, indym, indxp1, indypp1, indymp1
    cdef double wtlx, wtlyp, wtlym, wtrx, wtryp, wtrym
    
    cdef int i, j
    cdef int p, g, gee
    cdef double xp, yp
    
    cdef int nthread = openmp.omp_get_num_procs()
    
    cdef int indp, indp_xp1, indp_yp1, indp_xp1yp1
    cdef int indm, indm_xp1, indm_yp1, indm_xp1yp1
    
    cdef double [:,:] gridvals = np.zeros((ngx,ngy))
    
    for i in prange(ngx, nogil=True, num_threads=nthread):
        for j in range(ngy):
            xp = x[i]; yp = y[j]
            
            for g in range(ngrids-1):
                dxtmp = dxp[g]; dytmpp = dyp[g]
                ncelx = int(2**(g+1))
                ncely = int(2**(ngrids-g))
                ncelym = int(ncely/2)
                
                wtrx = xp/dxtmp; wtryp = yp/dytmpp; wtrym = 0.5*wtryp
                indx = <int> wtrx; indyp = <int> wtryp; indym = <int> wtrym
                wtrx = wtrx - indx; wtryp = wtryp - indyp; wtrym = wtrym - indym
                
                indxp1 = (indx+1) % ncelx; indypp1 = (indyp+1) % ncely; indymp1 = (indym+1) % ncelym
                
                wtlx = 1.0 - wtrx; wtlyp = 1.0 - wtryp; wtlym = 1.0 - wtrym
                
                indp = indx + indyp*ncelx; indp_xp1 = indxp1 + indyp*ncelx; indp_yp1 = indx + indypp1*ncelx; indp_xp1yp1 = indxp1 + indypp1*ncelx
                indm = indx + indym*ncelx; indm_xp1 = indxp1 + indym*ncelx; indm_yp1 = indx + indymp1*ncelx; indm_xp1yp1 = indxp1 + indymp1*ncelx
                
                gridvals[i,j] += wtlx*wtlyp*pgrids[g,indp] + wtrx*wtlyp*pgrids[g,indp_xp1] + wtlx*wtryp*pgrids[g,indp_yp1] + wtrx*wtryp*pgrids[g,indp_xp1yp1]
                gridvals[i,j] -= wtlx*wtlym*mgrids[g,indm] + wtrx*wtlym*mgrids[g,indm_xp1] + wtlx*wtrym*mgrids[g,indm_yp1] + wtrx*wtrym*mgrids[g,indm_xp1yp1]
                
            gee = ngrids-1
            ncelx = int(2**(gee+1))
            dxtmp = dxp[gee]
            dytmpp = dyp[gee]
            wtrx = xp/dxtmp; wtryp = yp/dytmpp
            indx = <int>wtrx; indyp = <int>wtryp
            wtrx = wtrx - indx; wtryp = wtryp - indyp
            indxp1 = (indx+1) % ncelx; indypp1 = (indyp+1) % 2
            
            wtlx = 1.0 - wtrx; wtlyp = 1.0 - wtryp
            indp = indx + indyp*ncelx; indp_xp1 = indxp1 + indyp*ncelx; indp_yp1 = indx + indypp1*ncelx; indp_xp1yp1 = indxp1 + indypp1*ncelx
            
            gridvals[i,j] += wtlx*wtlyp*pgrids[gee,indp] + wtrx*wtlyp*pgrids[gee,indp_xp1] + wtlx*wtryp*pgrids[gee,indp_yp1] + wtrx*wtryp*pgrids[gee,indp_xp1yp1]
        
    return np.asarray(gridvals)

###############################################################################
###############################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
def EvaluateOnGrid2Dbicubic(double[:] x, double[:] y, double[:,:] Fpgrids, double[:,:] Fmgrids, double[:] dxp, double [:] dyp):
    cdef int ngx = x.shape[0]
    cdef int ngy = y.shape[0]
    cdef int ngrids = Fpgrids.shape[0]
    cdef int ncelmax = Fmgrids.shape[1]
    
    cdef int ncelx, ncely, ncelym
    cdef double dxtmp, dytmpp, dytmpm
    
    cdef int indx, indyp, indym, indxp1, indypp1, indymp1, indxp2, indxm1, indypp2, indypm1, indymp2, indymm1
    cdef double wtrx, wtryp, wtrym, wx0, wy0p, wy0m, wx1, wy1p, wy1m, wx2, wy2p, wy2m, wxm1, wym1p, wym1m
    
    cdef int i, j
    cdef int p, g, gee
    cdef double xp, yp
    
    cdef int nthread = openmp.omp_get_num_procs()
    
    cdef int ip, ip_xp1, ip_yp1, ip_xp1yp1, ip_xp2, ip_xp2yp1, ip_xm1, 
    cdef int ip_xm1yp1, ip_yp2, ip_yp2xp1, ip_ym1, ip_ym1xp1, ip_xp2yp2, ip_xp2ym1, ip_xm1yp2, ip_xm1ym1
    cdef int im, im_xp1, im_yp1, im_xp1yp1, im_xp2, im_xp2yp1, im_xm1, 
    cdef int im_xm1yp1, im_yp2, im_yp2xp1, im_ym1, im_ym1xp1, im_xp2yp2, im_xp2ym1, im_xm1yp2, im_xm1ym1
    
    cdef double [:,:] gridvals = np.zeros((ngx,ngy))
    
    for i in prange(ngx, nogil=True, num_threads=nthread):
        for j in range(ngy):
            xp = x[i]; yp = y[j]
            
            for g in range(ngrids-1):
                dxtmp = dxp[g]; dytmpp = dyp[g]
                ncelx = int(2**(g+1))
                ncely = int(2**(ngrids-g))
                ncelym = int(ncely/2)
                
                wtrx = xp/dxtmp; wtryp = yp/dytmpp; wtrym = 0.5*wtryp
                indx = <int> wtrx; indyp = <int> wtryp; indym = <int> wtrym
                wtrx = wtrx - indx; wtryp = wtryp - indyp; wtrym = wtrym - indym
                
                indxp1 = (indx+1) % ncelx; indypp1 = (indyp+1) % ncely; indymp1 = (indym+1) % ncelym
                indxp2 = (indx+2) % ncelx; indxm1 = (indx-1) % ncelx; indypp2 = (indyp+2) % ncely; indypm1 = (indyp-1) % ncely
                indymp2 = (indym+2) % ncelym; indymm1 = (indym-1) % ncelym
                
                ip = indx + indyp*ncelx; ip_xp1 = indxp1 + indyp*ncelx; ip_yp1 = indx + indypp1*ncelx; ip_xp1yp1 = indxp1 + indypp1*ncelx
                ip_xp2 = indxp2 + indyp*ncelx; ip_xp2yp1 = indxp2 + indypp1*ncelx; ip_xm1 = indxm1 + indyp*ncelx; ip_xm1yp1 = indxm1 + indypp1*ncelx
                ip_yp2 = indx + indypp2*ncelx; ip_yp2xp1 = indxp1 + indypp2*ncelx; ip_ym1 = indx + indypm1*ncelx; ip_ym1xp1 = indxp1 + indypm1*ncelx
                ip_xp2yp2 = indxp2 + indypp2*ncelx; ip_xp2ym1 = indxp2 + indypm1*ncelx; ip_xm1yp2 = indxm1 + indypp2*ncelx; ip_xm1ym1 = indxm1 + indypm1*ncelx            
                
                im = indx + indym*ncelx; im_xp1 = indxp1 + indym*ncelx; im_yp1 = indx + indymp1*ncelx; im_xp1yp1 = indxp1 + indymp1*ncelx
                im_xp2 = indxp2 + indym*ncelx; im_xp2yp1 = indxp2 + indymp1*ncelx; im_xm1 = indxm1 + indym*ncelx; im_xm1yp1 = indxm1 + indymp1*ncelx
                im_yp2 = indx + indymp2*ncelx; im_yp2xp1 = indxp1 + indymp2*ncelx; im_ym1 = indx + indymm1*ncelx; im_ym1xp1 = indxp1 + indymm1*ncelx
                im_xp2yp2 = indxp2 + indymp2*ncelx; im_xp2ym1 = indxp2 + indymm1*ncelx; im_xm1yp2 = indxm1 + indymp2*ncelx; im_xm1ym1 = indxm1 + indymm1*ncelx
                
                wx0 = cubic0(wtrx); wy0p = cubic0(wtryp); wy0m = cubic0(wtrym); wx1 = cubic1(wtrx); wy1p = cubic1(wtryp); wy1m = cubic1(wtrym)
                wx2 = cubic2(wtrx); wy2p = cubic2(wtryp); wy2m = cubic2(wtrym); wxm1 = cubicm1(wtrx); wym1p = cubicm1(wtryp); wym1m = cubicm1(wtrym)
                
                gridvals[i,j] += wx0*wy0p*Fpgrids[g,ip] + wx1*wy0p*Fpgrids[g,ip_xp1] + wx0*wy1p*Fpgrids[g,ip_yp1] + wx1*wy1p*Fpgrids[g,ip_xp1yp1] + \
                    wx2*wy0p*Fpgrids[g,ip_xp2] + wx2*wy1p*Fpgrids[g,ip_xp2yp1] + wxm1*wy0p*Fpgrids[g,ip_xm1] + wxm1*wy1p*Fpgrids[g,ip_xm1yp1] + \
                    wx0*wy2p*Fpgrids[g,ip_yp2] + wx1*wy2p*Fpgrids[g,ip_yp2xp1] + wx0*wym1p*Fpgrids[g,ip_ym1] + wx1*wym1p*Fpgrids[g,ip_ym1xp1] + \
                    wx2*wy2p*Fpgrids[g,ip_xp2yp2] + wx2*wym1p*Fpgrids[g,ip_xp2ym1] + wxm1*wy2p*Fpgrids[g,ip_xm1yp2] + wxm1*wym1p*Fpgrids[g,ip_xm1ym1]
                gridvals[i,j] -= wx0*wy0m*Fmgrids[g,im] + wx1*wy0m*Fmgrids[g,im_xp1] + wx0*wy1m*Fmgrids[g,im_yp1] + wx1*wy1m*Fmgrids[g,im_xp1yp1] + \
                    wx2*wy0m*Fmgrids[g,im_xp2] + wx2*wy1m*Fmgrids[g,im_xp2yp1] + wxm1*wy0m*Fmgrids[g,im_xm1] + wxm1*wy1m*Fmgrids[g,im_xm1yp1] + \
                    wx0*wy2m*Fmgrids[g,im_yp2] + wx1*wy2m*Fmgrids[g,im_yp2xp1] + wx0*wym1m*Fmgrids[g,im_ym1] + wx1*wym1m*Fmgrids[g,im_ym1xp1] + \
                    wx2*wy2m*Fmgrids[g,im_xp2yp2] + wx2*wym1m*Fmgrids[g,im_xp2ym1] + wxm1*wy2m*Fmgrids[g,im_xm1yp2] + wxm1*wym1m*Fmgrids[g,im_xm1ym1]
                
            gee = ngrids-1
            g = gee
            ncelx = int(2**(gee+1)); ncely = 2
            dxtmp = dxp[gee]
            dytmpp = dyp[gee]
            wtrx = xp/dxtmp; wtryp = yp/dytmpp
            indx = <int>wtrx; indyp = <int>wtryp
            wtrx = wtrx - indx; wtryp = wtryp - indyp
            indxp1 = (indx+1) % ncelx; indypp1 = (indyp+1) % ncely
            indxp2 = (indx+2) % ncelx; indxm1 = (indx-1) % ncelx; indypp2 = (indyp+2) % ncely; indypm1 = (indyp-1) % ncely
            
            ip = indx + indyp*ncelx; ip_xp1 = indxp1 + indyp*ncelx; ip_yp1 = indx + indypp1*ncelx; ip_xp1yp1 = indxp1 + indypp1*ncelx
            ip_xp2 = indxp2 + indyp*ncelx; ip_xp2yp1 = indxp2 + indypp1*ncelx; ip_xm1 = indxm1 + indyp*ncelx; ip_xm1yp1 = indxm1 + indypp1*ncelx
            ip_yp2 = indx + indypp2*ncelx; ip_yp2xp1 = indxp1 + indypp2*ncelx; ip_ym1 = indx + indypm1*ncelx; ip_ym1xp1 = indxp1 + indypm1*ncelx
            ip_xp2yp2 = indxp2 + indypp2*ncelx; ip_xp2ym1 = indxp2 + indypm1*ncelx; ip_xm1yp2 = indxm1 + indypp2*ncelx; ip_xm1ym1 = indxm1 + indypm1*ncelx            
            
            wx0 = cubic0(wtrx); wy0p = cubic0(wtryp); wx1 = cubic1(wtrx); wy1p = cubic1(wtryp)
            wx2 = cubic2(wtrx); wy2p = cubic2(wtryp); wxm1 = cubicm1(wtrx); wym1p = cubicm1(wtryp)
            
            gridvals[i,j] += wx0*wy0p*Fpgrids[g,ip] + wx1*wy0p*Fpgrids[g,ip_xp1] + wx0*wy1p*Fpgrids[g,ip_yp1] + wx1*wy1p*Fpgrids[g,ip_xp1yp1] + \
                wx2*wy0p*Fpgrids[g,ip_xp2] + wx2*wy1p*Fpgrids[g,ip_xp2yp1] + wxm1*wy0p*Fpgrids[g,ip_xm1] + wxm1*wy1p*Fpgrids[g,ip_xm1yp1] + \
                wx0*wy2p*Fpgrids[g,ip_yp2] + wx1*wy2p*Fpgrids[g,ip_yp2xp1] + wx0*wym1p*Fpgrids[g,ip_ym1] + wx1*wym1p*Fpgrids[g,ip_ym1xp1] + \
                wx2*wy2p*Fpgrids[g,ip_xp2yp2] + wx2*wym1p*Fpgrids[g,ip_xp2ym1] + wxm1*wy2p*Fpgrids[g,ip_xm1yp2] + wxm1*wym1p*Fpgrids[g,ip_xm1ym1]
        
    return np.asarray(gridvals)
    
###############################################################################
###############################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def EvaluateAt(double[:,:] x, double[:,:] Fpgrids, double[:,:] Fmgrids, double [:] dxp, double [:] dyp):
    cdef int npar = x.shape[0]
    cdef int ngrids = Fpgrids.shape[0]
    cdef int ncelmax = Fmgrids.shape[1]
    cdef int n = int(np.log2(ncelmax))
    
    cdef double [:] Fpar = np.zeros(npar)
    
    cdef int ncelx, ncely, ncelym
    cdef double dxtmp, dytmpp, dytmpm
    
    cdef int indx, indyp, indym, indxp1, indypp1, indymp1
    cdef double wtlx, wtlyp, wtlym, wtrx, wtryp, wtrym
    
    cdef int p, g, gee
    cdef double xp, yp
    
    cdef int nthread = openmp.omp_get_num_procs()
    
    cdef int indp, indp_xp1, indp_yp1, indp_xp1yp1
    cdef int indm, indm_xp1, indm_yp1, indm_xp1yp1
    
    for p in prange(npar, nogil=True, num_threads=nthread):
        xp = x[p,0]; yp = x[p,1]

        for g in range(ngrids-1):
            dxtmp = dxp[g]; dytmpp = dyp[g]
            ncelx = int(2**(g+1))
            ncely = int(2**(ngrids-g))
            ncelym = int(ncely/2)
            
            wtrx = xp/dxtmp; wtryp = yp/dytmpp; wtrym = 0.5*wtryp
            indx = <int> wtrx; indyp = <int> wtryp; indym = <int> wtrym
            wtrx = wtrx - indx; wtryp = wtryp - indyp; wtrym = wtrym - indym
            
            indxp1 = (indx+1) % ncelx; indypp1 = (indyp+1) % ncely; indymp1 = (indym+1) % ncelym
            
            wtlx = 1.0 - wtrx; wtlyp = 1.0 - wtryp; wtlym = 1.0 - wtrym
            
            indp = indx + indyp*ncelx; indp_xp1 = indxp1 + indyp*ncelx; indp_yp1 = indx + indypp1*ncelx; indp_xp1yp1 = indxp1 + indypp1*ncelx
            indm = indx + indym*ncelx; indm_xp1 = indxp1 + indym*ncelx; indm_yp1 = indx + indymp1*ncelx; indm_xp1yp1 = indxp1 + indymp1*ncelx
            
            Fpar[p] += wtlx*wtlyp*Fpgrids[g,indp] + wtrx*wtlyp*Fpgrids[g,indp_xp1] + wtlx*wtryp*Fpgrids[g,indp_yp1] + wtrx*wtryp*Fpgrids[g,indp_xp1yp1]
            Fpar[p] -= wtlx*wtlym*Fmgrids[g,indm] + wtrx*wtlym*Fmgrids[g,indm_xp1] + wtlx*wtrym*Fmgrids[g,indm_yp1] + wtrx*wtrym*Fmgrids[g,indm_xp1yp1]
            
        gee = ngrids-1
        ncelx = int(2**(gee+1)); ncely = 2
        dxtmp = dxp[gee]
        dytmpp = dyp[gee]
        wtrx = xp/dxtmp; wtryp = yp/dytmpp
        indx = <int>wtrx; indyp = <int>wtryp
        wtrx = wtrx - indx; wtryp = wtryp - indyp
        indxp1 = (indx+1) % ncelx; indypp1 = (indyp+1) % ncely
        
        wtlx = 1.0 - wtrx; wtlyp = 1.0 - wtryp
        indp = indx + indyp*ncelx; indp_xp1 = indxp1 + indyp*ncelx; indp_yp1 = indx + indypp1*ncelx; indp_xp1yp1 = indxp1 + indypp1*ncelx
        
        Fpar[p] += wtlx*wtlyp*Fpgrids[gee,indp] + wtrx*wtlyp*Fpgrids[gee,indp_xp1] + wtlx*wtryp*Fpgrids[gee,indp_yp1] + wtrx*wtryp*Fpgrids[gee,indp_xp1yp1]
        
    return np.asarray(Fpar)

###############################################################################
###############################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
def EvaluateAtBicubic(double[:,:] x, double[:,:] Fpgrids, double[:,:] Fmgrids, double [:] dxp, double [:] dyp):
    cdef int npar = x.shape[0]
    cdef int ngrids = Fpgrids.shape[0]
    cdef int ncelmax = Fmgrids.shape[1]
    cdef int n = int(np.log2(ncelmax))
    
    cdef double [:] Fpar = np.zeros(npar)
    
    cdef int ncelx, ncely, ncelym
    cdef double dxtmp, dytmpp, dytmpm
    
    cdef int indx, indyp, indym, indxp1, indypp1, indymp1, indxp2, indxm1, indypp2, indypm1, indymp2, indymm1
    cdef double wtrx, wtryp, wtrym, wx0, wy0p, wy0m, wx1, wy1p, wy1m, wx2, wy2p, wy2m, wxm1, wym1p, wym1m
    
    cdef int p, g, gee
    cdef double xp, yp
    
    cdef int nthread = openmp.omp_get_num_procs()
    
    cdef int ip, ip_xp1, ip_yp1, ip_xp1yp1, ip_xp2, ip_xp2yp1, ip_xm1, 
    cdef int ip_xm1yp1, ip_yp2, ip_yp2xp1, ip_ym1, ip_ym1xp1, ip_xp2yp2, ip_xp2ym1, ip_xm1yp2, ip_xm1ym1
    cdef int im, im_xp1, im_yp1, im_xp1yp1, im_xp2, im_xp2yp1, im_xm1, 
    cdef int im_xm1yp1, im_yp2, im_yp2xp1, im_ym1, im_ym1xp1, im_xp2yp2, im_xp2ym1, im_xm1yp2, im_xm1ym1
    
    for p in prange(npar, nogil=True, num_threads=nthread):
        xp = x[p,0]; yp = x[p,1]

        for g in range(ngrids-1):
            dxtmp = dxp[g]; dytmpp = dyp[g]
            ncelx = int(2**(g+1))
            ncely = int(2**(ngrids-g))
            ncelym = int(ncely/2)
            
            wtrx = xp/dxtmp; wtryp = yp/dytmpp; wtrym = 0.5*wtryp
            indx = <int> wtrx; indyp = <int> wtryp; indym = <int> wtrym
            wtrx = wtrx - indx; wtryp = wtryp - indyp; wtrym = wtrym - indym
            
            indxp1 = (indx+1) % ncelx; indypp1 = (indyp+1) % ncely; indymp1 = (indym+1) % ncelym
            indxp2 = (indx+2) % ncelx; indxm1 = (indx-1) % ncelx; indypp2 = (indyp+2) % ncely; indypm1 = (indyp-1) % ncely
            indymp2 = (indym+2) % ncelym; indymm1 = (indym-1) % ncelym
            
            ip = indx + indyp*ncelx; ip_xp1 = indxp1 + indyp*ncelx; ip_yp1 = indx + indypp1*ncelx; ip_xp1yp1 = indxp1 + indypp1*ncelx
            ip_xp2 = indxp2 + indyp*ncelx; ip_xp2yp1 = indxp2 + indypp1*ncelx; ip_xm1 = indxm1 + indyp*ncelx; ip_xm1yp1 = indxm1 + indypp1*ncelx
            ip_yp2 = indx + indypp2*ncelx; ip_yp2xp1 = indxp1 + indypp2*ncelx; ip_ym1 = indx + indypm1*ncelx; ip_ym1xp1 = indxp1 + indypm1*ncelx
            ip_xp2yp2 = indxp2 + indypp2*ncelx; ip_xp2ym1 = indxp2 + indypm1*ncelx; ip_xm1yp2 = indxm1 + indypp2*ncelx; ip_xm1ym1 = indxm1 + indypm1*ncelx            
            
            im = indx + indym*ncelx; im_xp1 = indxp1 + indym*ncelx; im_yp1 = indx + indymp1*ncelx; im_xp1yp1 = indxp1 + indymp1*ncelx
            im_xp2 = indxp2 + indym*ncelx; im_xp2yp1 = indxp2 + indymp1*ncelx; im_xm1 = indxm1 + indym*ncelx; im_xm1yp1 = indxm1 + indymp1*ncelx
            im_yp2 = indx + indymp2*ncelx; im_yp2xp1 = indxp1 + indymp2*ncelx; im_ym1 = indx + indymm1*ncelx; im_ym1xp1 = indxp1 + indymm1*ncelx
            im_xp2yp2 = indxp2 + indymp2*ncelx; im_xp2ym1 = indxp2 + indymm1*ncelx; im_xm1yp2 = indxm1 + indymp2*ncelx; im_xm1ym1 = indxm1 + indymm1*ncelx
            
            wx0 = cubic0(wtrx); wy0p = cubic0(wtryp); wy0m = cubic0(wtrym); wx1 = cubic1(wtrx); wy1p = cubic1(wtryp); wy1m = cubic1(wtrym)
            wx2 = cubic2(wtrx); wy2p = cubic2(wtryp); wy2m = cubic2(wtrym); wxm1 = cubicm1(wtrx); wym1p = cubicm1(wtryp); wym1m = cubicm1(wtrym)
            
            Fpar[p] += wx0*wy0p*Fpgrids[g,ip] + wx1*wy0p*Fpgrids[g,ip_xp1] + wx0*wy1p*Fpgrids[g,ip_yp1] + wx1*wy1p*Fpgrids[g,ip_xp1yp1] + \
                wx2*wy0p*Fpgrids[g,ip_xp2] + wx2*wy1p*Fpgrids[g,ip_xp2yp1] + wxm1*wy0p*Fpgrids[g,ip_xm1] + wxm1*wy1p*Fpgrids[g,ip_xm1yp1] + \
                wx0*wy2p*Fpgrids[g,ip_yp2] + wx1*wy2p*Fpgrids[g,ip_yp2xp1] + wx0*wym1p*Fpgrids[g,ip_ym1] + wx1*wym1p*Fpgrids[g,ip_ym1xp1] + \
                wx2*wy2p*Fpgrids[g,ip_xp2yp2] + wx2*wym1p*Fpgrids[g,ip_xp2ym1] + wxm1*wy2p*Fpgrids[g,ip_xm1yp2] + wxm1*wym1p*Fpgrids[g,ip_xm1ym1]
            Fpar[p] -= wx0*wy0m*Fmgrids[g,im] + wx1*wy0m*Fmgrids[g,im_xp1] + wx0*wy1m*Fmgrids[g,im_yp1] + wx1*wy1m*Fmgrids[g,im_xp1yp1] + \
                wx2*wy0m*Fmgrids[g,im_xp2] + wx2*wy1m*Fmgrids[g,im_xp2yp1] + wxm1*wy0m*Fmgrids[g,im_xm1] + wxm1*wy1m*Fmgrids[g,im_xm1yp1] + \
                wx0*wy2m*Fmgrids[g,im_yp2] + wx1*wy2m*Fmgrids[g,im_yp2xp1] + wx0*wym1m*Fmgrids[g,im_ym1] + wx1*wym1m*Fmgrids[g,im_ym1xp1] + \
                wx2*wy2m*Fmgrids[g,im_xp2yp2] + wx2*wym1m*Fmgrids[g,im_xp2ym1] + wxm1*wy2m*Fmgrids[g,im_xm1yp2] + wxm1*wym1m*Fmgrids[g,im_xm1ym1]
            
        gee = ngrids-1
        ncelx = int(2**(gee+1)); ncely = 2
        dxtmp = dxp[gee]
        dytmpp = dyp[gee]
        wtrx = xp/dxtmp; wtryp = yp/dytmpp
        indx = <int>wtrx; indyp = <int>wtryp
        wtrx = wtrx - indx; wtryp = wtryp - indyp
        indxp1 = (indx+1) % ncelx; indypp1 = (indyp+1) % ncely
        indxp2 = (indx+2) % ncelx; indxm1 = (indx-1) % ncelx; indypp2 = (indyp+2) % ncely; indypm1 = (indyp-1) % ncely
            
        ip = indx + indyp*ncelx; ip_xp1 = indxp1 + indyp*ncelx; ip_yp1 = indx + indypp1*ncelx; ip_xp1yp1 = indxp1 + indypp1*ncelx
        ip_xp2 = indxp2 + indyp*ncelx; ip_xp2yp1 = indxp2 + indypp1*ncelx; ip_xm1 = indxm1 + indyp*ncelx; ip_xm1yp1 = indxm1 + indypp1*ncelx
        ip_yp2 = indx + indypp2*ncelx; ip_yp2xp1 = indxp1 + indypp2*ncelx; ip_ym1 = indx + indypm1*ncelx; ip_ym1xp1 = indxp1 + indypm1*ncelx
        ip_xp2yp2 = indxp2 + indypp2*ncelx; ip_xp2ym1 = indxp2 + indypm1*ncelx; ip_xm1yp2 = indxm1 + indypp2*ncelx; ip_xm1ym1 = indxm1 + indypm1*ncelx            
        
        
        wx0 = cubic0(wtrx); wy0p = cubic0(wtryp); wx1 = cubic1(wtrx); wy1p = cubic1(wtryp)
        wx2 = cubic2(wtrx); wy2p = cubic2(wtryp); wxm1 = cubicm1(wtrx); wym1p = cubicm1(wtryp)
        
        g = gee        
        
        Fpar[p] += wx0*wy0p*Fpgrids[g,ip] + wx1*wy0p*Fpgrids[g,ip_xp1] + wx0*wy1p*Fpgrids[g,ip_yp1] + wx1*wy1p*Fpgrids[g,ip_xp1yp1] + \
            wx2*wy0p*Fpgrids[g,ip_xp2] + wx2*wy1p*Fpgrids[g,ip_xp2yp1] + wxm1*wy0p*Fpgrids[g,ip_xm1] + wxm1*wy1p*Fpgrids[g,ip_xm1yp1] + \
            wx0*wy2p*Fpgrids[g,ip_yp2] + wx1*wy2p*Fpgrids[g,ip_yp2xp1] + wx0*wym1p*Fpgrids[g,ip_ym1] + wx1*wym1p*Fpgrids[g,ip_ym1xp1] + \
            wx2*wy2p*Fpgrids[g,ip_xp2yp2] + wx2*wym1p*Fpgrids[g,ip_xp2ym1] + wxm1*wy2p*Fpgrids[g,ip_xm1yp2] + wxm1*wym1p*Fpgrids[g,ip_xm1ym1]
               
        
    return np.asarray(Fpar)