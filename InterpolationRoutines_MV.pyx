# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 18:11:42 2016

@author: ricketsonl
"""
# cython: profile=False
# cython: linetrace=False
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange, threadid
cimport openmp
@cython.boundscheck(False)
@cython.cdivision(True)

## Intepolate a 1-D periodic function onto a grid with twice the resolution #########################################
def LinInterpPeriodic(np.ndarray[double, ndim=1] P):
    cdef unsigned int Psize = P.shape[0]
    cdef np.ndarray[np.double_t,ndim=1] Pint
    Pint = np.zeros(2*Psize,dtype=np.double)
    
    for i in range(Psize-1):
        Pint[2*i] = P[i]
        Pint[2*i+1] = 0.5*(P[i]+P[i+1])
    Pint[2*(Psize-1)] = P[Psize-1]
    Pint[2*Psize-1] = 0.5*(P[Psize-1] + P[0])
    
    return Pint

## Interpolate 1-D particle data onto a uniform 1-D grid ##########################################
def InterpGrid(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] v, double dx, unsigned int ncel):
    cdef unsigned int i, ind
    #cdef double Len = ncel*dx
    cdef double wtl, wtr, xl
    cdef np.ndarray[np.double_t,ndim=1] F
    cdef unsigned int xsize = x.shape[0]
    F = np.zeros(ncel,dtype=np.double)
    
    for i in range(xsize):
        ind = <unsigned int>(x[i]/dx)
        xl = ind*dx
        wtl = (x[i]-xl)/dx; wtr = 1.0-wtl
        F[ind] += wtl*v[i]
        
        if ind < ncel-1:
            F[ind+1] += wtr*v[i]
        else:
            F[0] += wtr*v[i]
            
    return F/(dx*xsize)
    

def InterpPar(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] F, double dx, unsigned int ncel):
    cdef unsigned int i, ind
    cdef double wtl, wtr, xl
    cdef np.ndarray[np.double_t,ndim=1] v
    cdef unsigned int xsize = x.shape[0]
    v = np.zeros(xsize,dtype=np.double)
    
    for i in range(xsize):
        ind = <unsigned int>(x[i]/dx)
        xl = ind*dx
        wtl = (x[i]-xl)/dx; wtr = 1.0-wtl
        v[i] += wtl*F[ind]
        if ind < ncel-1:
            v[i] += wtr*F[ind+1]
        else:
            v[i] += wtr*F[0]
            
    return v
    
def Var0(np.ndarray[double, ndim=1] x, double dx, unsigned int ncel):
    cdef unsigned int i, ind
    #cdef double Len = ncel*dx
    cdef double wtl, wtr, xl
    cdef np.ndarray[np.double_t,ndim=1] V
    cdef unsigned int xsize = x.shape[0]
    V = np.zeros(ncel,dtype=np.double)
    
    for i in range(xsize):
        ind = <unsigned int>(x[i]/dx)
        xl = ind*dx
        wtl = (x[i]-xl)/dx; wtr = 1.0-wtl
        V[ind] += wtl*wtl
        if ind < ncel-1:
            V[ind+1] += wtr*wtr
        else:
            V[0] += wtr*wtr
    
    return V/(dx*dx*xsize)
    
def Varl(np.ndarray[double, ndim=1] xf, np.ndarray[double, ndim=1] xc, double dx, unsigned int ncel):
    cdef unsigned int i, indf, indc, indfp1, indcp1
    #cdef double Len = ncel*dx
    cdef double wtlc, wtlf, wtrc, wtrf, xlf, xlc
    cdef np.ndarray[np.double_t,ndim=1] V
    cdef unsigned int xsize = xf.shape[0]
    V = np.zeros(ncel,dtype=np.double)
    for i in range(xsize):
        indf = <unsigned int>(xf[i]/dx); indc = <unsigned int>(xc[i]/dx)
        indfp1 = (indf+1) % ncel; indcp1 = (indc+1) % ncel
        xlf = indf*dx; xlc = indc*dx
        wtlf = (xf[i]-xlf)/dx; wtrf = 1.0 - wtlf
        wtlc = (xc[i]-xlc)/dx; wtrc = 1.0 - wtlc
        if indc == indf:
            V[indf] += (wtlf-wtlc)*(wtlf-wtlc)
            V[indfp1] += (wtrf-wtrc)*(wtrf-wtrc)
        elif indc == indfp1:
            V[indc] += (wtrf-wtlc)*(wtrf - wtlc)
            V[indf] += wtlf*wtlf
            V[indcp1] += wtrc*wtrc
        elif indf == indcp1:
            V[indf] += (wtlf-wtrc)*(wtlf-wtrc)
            V[indc] += wtlc*wtlc
            V[indfp1] += wtrf*wtrf
        else:
            V[indf] += wtlf*wtlf; V[indfp1] += wtrf*wtrf
            V[indc] += wtlc*wtlc; V[indcp1] += wtrc*wtrc
            
    return V/(dx*dx*xsize)
    
######### 2-D Routines #########
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def InterpGrid2D(double [:,:] x, double [:] v, double dx, double dy, unsigned int ncelx, unsigned int ncely, double [:,:] F):
    cdef unsigned int i, j, k, indx, indxp1, indy, indyp1, tid
    cdef double wtxl, wtxr, wtyl, wtyr, val
    cdef unsigned int xsize = x.shape[0]
    
    cdef unsigned int nthread = openmp.omp_get_num_procs()
    cdef double [:,:,:] Fbig = np.zeros((ncelx,ncely,nthread))
    cdef double div = dx*dy*xsize
    
    for i in prange(xsize, nogil=True, num_threads=nthread):
        tid = threadid()
        wtxr = x[i,0]/dx; wtyr = x[i,1]/dy
        indx = <unsigned int> wtxr; indy = <unsigned int> wtyr
        wtxr = wtxr - indx; wtyr = wtyr - indy
        wtxl = 1.0 - wtxr; wtyl = 1.0 - wtyr  
        
        val = v[i]
        wtxl = wtxl*val; wtxr = wtxr*val
        Fbig[indx,indy,tid] += wtxl*wtyl
        indxp1 = (indx+1) % ncelx; indyp1 = (indy+1) % ncely
        Fbig[indxp1,indy,tid] += wtxr*wtyl
        Fbig[indx,indyp1,tid] += wtxl*wtyr
        Fbig[indxp1,indyp1,tid] += wtxr*wtyr
        
    for i in range(ncelx):
        for j in range(ncely):
            for k in range(nthread):
                F[i,j] += Fbig[i,j,k]
            F[i,j] /= div
            
cdef inline double QuarticDeltaApprox(double x) nogil:
    return 1.40625 - 4.6875*x*x + 3.28125*x*x*x*x

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)  
def InterpGrid2Dbicubic(double [:,:] x, double [:] v, double dx, double dy, unsigned int ncelx, unsigned int ncely, double [:,:] F):
    cdef unsigned int i, j, k, indx, indxp1, indy, indyp1, tid
    cdef double wtxl, wtxr, wtyl, wtyr, val
    cdef unsigned int xsize = x.shape[0]
    
    cdef unsigned int nthread = openmp.omp_get_num_procs()
    cdef double [:,:,:] Fbig = np.zeros((ncelx,ncely,nthread))
    cdef double div = dx*dy*xsize
    
    for i in prange(xsize, nogil=True, num_threads=nthread):
        tid = threadid()
        wtxr = x[i,0]/dx; wtyr = x[i,1]/dy
        indx = <unsigned int> wtxr; indy = <unsigned int> wtyr
        wtxr = wtxr - indx; wtyr = wtyr - indy
        wtxl = 1.0 - wtxr; wtyl = 1.0 - wtyr  
        
        wtxl = QuarticDeltaApprox(1.0-wtxl); wtxr = QuarticDeltaApprox(1.0-wtxr)
        wtyl = QuarticDeltaApprox(1.0-wtyl); wtyr = QuarticDeltaApprox(1.0-wtyr)
        
        val = v[i]
        wtxl = wtxl*val; wtxr = wtxr*val
        Fbig[indx,indy,tid] += wtxl*wtyl
        indxp1 = (indx+1) % ncelx; indyp1 = (indy+1) % ncely
        Fbig[indxp1,indy,tid] += wtxr*wtyl
        Fbig[indx,indyp1,tid] += wtxl*wtyr
        Fbig[indxp1,indyp1,tid] += wtxr*wtyr
        
    for i in range(ncelx):
        for j in range(ncely):
            for k in range(nthread):
                F[i,j] += Fbig[i,j,k]
            F[i,j] /= div
        
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def InterpPar2D(double [:,:] x, double [:,:] F, double dx, double dy, unsigned int ncelx, unsigned int ncely):
    cdef unsigned int i, indx, indxp1, indy, indyp1
    cdef double wtxl, wtxr, wtyl, wtyr
    cdef unsigned int xsize = x.shape[0]
    cdef double [:] Fpar = np.zeros(xsize)
    
    for i in prange(xsize, nogil=True):
        wtxr = x[i,0]/dx; wtyr = x[i,1]/dy
        indx = <unsigned int> wtxr; indy = <unsigned int> wtyr
        wtxr = wtxr - indx; wtyr = wtyr - indy
        wtxl = 1.0-wtxr; wtyl = 1.0-wtyr
        indxp1 = (indx+1) % ncelx; indyp1 = (indy+1) % ncely
        Fpar[i] = wtxl*wtyl*F[indx,indy] + wtxr*wtyl*F[indxp1,indy] + wtxl*wtyr*F[indx,indyp1] + wtxr*wtyr*F[indxp1,indyp1]
    
    return np.asarray(Fpar)
  
cdef inline double cubic0(double x) nogil:
    return (x-1.0)*(x+1.0)*(x-2.0)/2.0

cdef inline double cubicp1(double x) nogil:
    return x*(2.0-x)*(1.0+x)/2.0

cdef inline double cubicp2(double x) nogil:
    return x*(x-1.0)*(x+1.0)/6.0

cdef inline double cubicm1(double x) nogil:
    return x*(x-1.0)*(2.0-x)/6.0    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def InterpPar2Dbicubic(double[:,:] x, double[:,:] F, double dx, double dy, unsigned int ncelx, unsigned int ncely):
    cdef unsigned int i, indx, indxp1, indxp2, indxm1, indy, indyp1, indyp2, indym1
    cdef double wtxr, wtyr, wx0, wy0, wx1, wy1, wx2, wy2, wxm1, wym1
    cdef unsigned int xsize = x.shape[0]
    cdef double[:] Fpar = np.zeros(xsize)
    
    for i in prange(xsize, nogil=True):
        wtxr = x[i,0]/dx; wtyr = x[i,1]/dy
        indx = <unsigned int> wtxr; indy = <unsigned int> wtyr
        wtxr = wtxr - indx; wtyr = wtyr - indy
        #wtxl = 1.0-wtxr; wtyl = 1.0-wtyr
        indxp1 = (indx+1) % ncelx; indyp1 = (indy+1) % ncely; indxp2 = (indx+2) % ncelx; indyp2 = (indy+2) % ncely
        indxm1 = (indx-1) % ncelx; indym1 = (indy-1) % ncely
        wx0 = cubic0(wtxr); wy0 = cubic0(wtyr); wx1 = cubicp1(wtxr); wy1 = cubicp1(wtyr); wx2 = cubicp2(wtxr); wy2 = cubicp2(wtyr)
        wxm1 = cubicm1(wtxr); wym1 = cubicm1(wtyr)
        Fpar[i] = wx0*wy0*F[indx,indy] + wx1*wy0*F[indxp1,indy] + wx0*wy1*F[indx,indyp1] + wx1*wy1*F[indxp1,indyp1] + \
            wx2*wy0*F[indxp2,indy] + wx2*wy1*F[indxp2,indyp1] + wxm1*wy0*F[indxm1,indy] + wxm1*wy1*F[indxm1,indyp1] + \
            wx0*wy2*F[indx,indyp2] + wx1*wy2*F[indxp1,indyp2] + wx0*wym1*F[indx,indym1] + wx1*wym1*F[indxp1,indym1] + \
            wx2*wy2*F[indxp2,indyp2] + wx2*wym1*F[indxp2,indym1] + wxm1*wy2*F[indxm1,indyp2] + wxm1*wym1*F[indxm1,indym1]
            
    return np.asarray(Fpar)

######### 3-D Routines #########
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def InterpGrid3D(double [:,:] x, double [:] v, double dx, double dy, double dz, unsigned int ncelx, unsigned int ncely, unsigned int ncelz, double [:,:,:] F):
    cdef unsigned int i, j, k, l, indx, indy, indz, indxp1, indyp1, indzp1, tid
    cdef double wtxl, wtxr, wtyl, wtyr, wtzl, wtzr, val
    cdef unsigned int xsize = x.shape[0]

    cdef unsigned int nthread = openmp.omp_get_num_procs()
    cdef double [:,:,:,:] Fbig = np.zeros((ncelx,ncely,ncelz,nthread))
    cdef double div = dx*dy*dz*xsize
    
    for i in prange(xsize, nogil=True):
        tid = threadid()
        wtxr = x[i,0]/dx; wtyr = x[i,1]/dy; wtzr = x[i,2]/dz
        indx = <unsigned int> wtxr; indy = <unsigned int> wtyr; indz = <unsigned int> wtzr
        wtxr = wtxr - indx; wtyr = wtyr - indy; wtzr = wtzr - indz
        wtxl = 1.0 - wtxr; wtyl = 1.0 - wtyr; wtzl = 1.0 - wtzr
        
        indxp1 = (indx+1) % ncelx; indyp1 = (indy+1) % ncely; indzp1 = (indz+1) % ncelz

        val = v[i]; wtxl = wtxl*val; wtxr = wtxr*val

        Fbig[indx,indy,indz,tid] += wtxl*wtyl*wtzl
        Fbig[indxp1,indy,indz,tid] += wtxr*wtyl*wtzl
        Fbig[indx,indyp1,indz,tid] += wtxl*wtyr*wtzl
        Fbig[indx,indy,indzp1,tid] += wtxl*wtyl*wtzr
        Fbig[indxp1,indyp1,indz,tid] += wtxr*wtyr*wtzl
        Fbig[indxp1,indy,indzp1,tid] += wtxr*wtyl*wtzr
        Fbig[indx,indyp1,indzp1,tid] += wtxl*wtyr*wtzr
        Fbig[indxp1,indyp1,indzp1,tid] += wtxr*wtyr*wtzr
        
    for i in range(ncelx):
        for j in range(ncely):
            for k in range(ncelz):
                for l in range(nthread):
                    F[i,j,k] += Fbig[i,j,k,l]
                F[i,j,k] /= div
    return F
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def InterpPar3D(double [:,:] x, double [:,:,:] F, double dx, double dy, double dz, unsigned int ncelx, unsigned int ncely, unsigned int ncelz):
    cdef unsigned int i, indx, indy, indz, indxp1, indyp1, indzp1
    cdef double wtxl, wtxr, wtyl, wtyr, wtzl, wtzr
    cdef unsigned int xsize = x.shape[0]
    cdef double [:] Fpar = np.zeros(xsize)
    
    for i in prange(xsize, nogil=True):
        wtxr = x[i,0]/dx; wtyr = x[i,1]/dy; wtzr = x[i,2]/dz
        indx = <unsigned int> wtxr; indy = <unsigned int> wtyr; indz = <unsigned int> wtzr
        wtxr = wtxr - indx; wtyr = wtyr - indy; wtzr = wtzr - indz
        wtxl = 1.0 - wtxr; wtyl = 1.0 - wtyr; wtzl = 1.0 - wtzr
        
        indxp1 = (indx+1) % ncelx; indyp1 = (indy+1) % ncely; indzp1 = (indz+1) % ncelz

        Fpar[i] = wtxl*wtyl*wtzl*F[indx,indy,indz] + wtxr*wtyl*wtzl*F[indxp1,indy,indz] + wtxl*wtyr*wtzl*F[indx,indyp1,indz]+wtxl*wtyl*wtzr*F[indx,indy,indzp1] + wtxr*wtyr*wtzl*F[indxp1,indyp1,indz] + wtxr*wtyl*wtzr*F[indxp1,indy,indzp1]+wtxl*wtyr*wtzr*F[indx,indyp1,indzp1] + wtxr*wtyr*wtzr*F[indxp1,indyp1,indzp1]
    
    return np.asarray(Fpar) 