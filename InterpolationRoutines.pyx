# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:46:31 2016

@author: lfr
"""
import numpy as np
cimport numpy as np
cimport cython
@cython.boundscheck(False)
@cython.cdivision(True)

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
    
def SpaceVar(np.ndarray[double, ndim=1] xf, np.ndarray[double, ndim=1] xc, double dxc, unsigned int ncelc):
    cdef unsigned int i, indf, indc, indfp1, indfp2, indfp3, indfm1, indfm2, indfm3    
    cdef unsigned int ncelf = 2*ncelc
    cdef double dxf = 0.5*dxc
    cdef double wtlc, wtlf, wtrc, wtrf, xlf, xlc
    cdef np.ndarray[np.double_t,ndim=1] V
    cdef unsigned int xsize = xf.shape[0]
    V = np.zeros(ncelf,dtype=np.double)
    
    # Everything below assumes at least 4 coarse cells (8 fine cells), o/w "elif" should be "if" due to wrapping
    for i in range(xsize): 
        indf = <unsigned int>(xf[i]/dxf); indc = <unsigned int>(xc[i]/dxc)
        xlf = indf*dxf; xlc = indc*dxc
        indfp1 = (indf+1) % ncelf; indfm1 = (indf-1) % ncelf
        indfp2 = (indf+2) % ncelf; indfp3 = (indf+3) % ncelf
        indfm2 = (indf-2) % ncelf; indfm3 = (indf-3) % ncelf
        wtlf = (xf[i]-xlf)/dxf; wtrf = 1.0 - wtlf
        wtlc = (xc[i]-xlc)/dxc; wtrc = 1.0 - wtlc
        
        if indf == 2*indc: # Same coarse cell, fine particle is in left half
            V[indf] += (wtlf/dxf - wtlc/dxc)*(wtlf/dxf - wtlc/dxc)
            V[indfp1] += (wtrf/dxf - 0.5/dxc)*(wtrf/dxf - 0.5/dxc)
            V[indfp2] += wtrc*wtrc/(dxc*dxc)
            V[indfp3] += 0.25*wtrc*wtrc/(dxc*dxc)
            V[indfm1] += 0.25*wtlc*wtlf/(dxc*dxc)
        elif indf == (2*indc+1) % ncelf: # Same coarse cell, fine particle is in right half
            V[indf] += (wtlf/dxf - 0.5/dxc)*(wtlf/dxf - 0.5/dxc)
            V[indfp1] += (wtrf/dxf - wtrc/dxc)*(wtrf/dxf - wtrc/dxc)
            V[indfp2] += 0.25*wtrc*wtrc/(dxc*dxc)
            V[indfm1] += wtlc*wtlc/(dxc*dxc)
            V[indfm2] += 0.25*wtlc*wtlc/(dxc*dxc)
        elif indf == (2*indc-1) % ncelf: # C par in c cell to right of f par's c cell, f par in r half of its c cell
            V[indfp1] += (wtrf/dxf - wtlc/dxc)*(wtrf/dxf - wtlc/dxc)
            V[indf] += (wtlf/dxf - 0.5*wtlc/dxc)*(wtlf/dxf - 0.5*wtlc/dxc)
            V[indfp2] += 0.25/(dxc*dxc)
            V[indfp3] += wtrc*wtrc/(dxc*dxc)
            V[(indf+4) % ncelf] += 0.25*wtrc*wtrc/(dxc*dxc)
        elif indf == (2*indc-2) % ncelf: # C par in c cell to right of f par's c cell, f par in r half of its c cell
            V[indfp1] += (wtrf/dxf - 0.5*wtlc/dxc)*(wtrf/dxf - 0.5*wtlc/dxc)
            V[indf] += wtlf*wtlf/(dxf*dxf)
            V[indfp2] += wtlc*wtlc/(dxc*dxc)
            V[indfp3] += 0.25/(dxc*dxc)
            V[(indf+4) % ncelf] += wtrc*wtrc/(dxc*dxc)
            V[(indf+5) % ncelf] += 0.25*wtrc*wtrc/(dxc*dxc)
        elif indf == (2*indc+2) % ncelf: # C par in c cell to left of f par's c cell, f par in l half of its c cell
            V[indf] += (wtlf/dxf - wtrc/dxc)*(wtlf/dxf - wtrc/dxc)
            V[indfp1] += (wtrf/dxf - 0.5*wtrc/dxc)*(wtrf/dxf - 0.5*wtrc/dxc)
            V[indfm1] += 0.25/(dxc*dxc)
            V[indfm2] += wtlc*wtlc/(dxc*dxc)
            V[indfm3] += 0.25*wtlc*wtlc/(dxc*dxc)
        elif indf == (2*indc+3) % ncelf: # C par in c cell to left of f par's c cell, f par in r half of its c cell
            V[indf] += (wtlf/dxf - 0.5*wtrc/dxc)*(wtlf/dxf - 0.5*wtrc/dxc)
            V[indfp1] += wtrf*wtrf/(dxc*dxc)
            V[indfm1] += wtrc*wtrc/(dxc*dxc)
            V[indfm2] += 0.25/(dxc*dxc)
            V[indfm3] += wtlc*wtlc/(dxc*dxc)
            V[(indf-4) % ncelf] += 0.25*wtlc*wtlc
        else: # Nowhere near each other
            V[indf] += wtlf*wtlf/(dxf*dxf)
            V[indfp1] += wtrf*wtrf/(dxf*dxf)
            V[2*indc] += wtlc*wtlc/(dxc*dxc)
            V[(2*indc+1) % ncelf] += 0.25/(dxc*dxc)
            V[(2*indc+2) % ncelf] += wtrc*wtrc/(dxc*dxc)
            V[(2*indc+3) % ncelf] += 0.25*wtrc*wtrc/(dxc*dxc)
            V[(2*indc-1) % ncelf] += 0.25*wtlc*wtlc/(dxf*dxf)
            
    return V/xsize
    
######### 2-D Routines #########   
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def InterpGrid2D(np.ndarray[double, ndim=2] x, np.ndarray[double, ndim=1] v, double dx, double dy, unsigned int ncelx, unsigned int ncely):
    cdef unsigned int i, indx, indxp1, indy, indyp1
    cdef double wtxl, wtxr, wtyl, wtyr, val
    cdef np.ndarray[np.double_t,ndim=2] F
    cdef unsigned int xsize = x.shape[0]
    F = np.zeros((ncelx,ncely),dtype=np.double)
    
    for i in range(xsize):
        
        wtxr = x[i,0]/dx; wtyr = x[i,1]/dy
        indx = <unsigned int> wtxr; indy = <unsigned int> wtyr
        wtxr -= indx; wtyr -= indy
        wtxl = 1.0 - wtxr; wtyl = 1.0 - wtyr   
        
        val = v[i]
        wtxl *= val; wtxr *= val
        F[indx,indy] += wtxl*wtyl
        
        ## No if-statement method
        indxp1 = (indx+1) % ncelx; indyp1 = (indy+1) % ncely
        F[indxp1,indy] += wtxr*wtyl
        F[indx,indyp1] += wtxl*wtyr
        F[indxp1,indyp1] += wtxr*wtyr
        
    return F/(dx*dy*xsize)
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def InterpDensGrid2D(np.ndarray[double, ndim=2] x, double dx, double dy, unsigned int ncelx, unsigned int ncely):
    cdef unsigned int i, indx, indxp1, indy, indyp1
    cdef double wtxl, wtxr, wtyl, wtyr
    cdef np.ndarray[np.double_t,ndim=2] F
    cdef unsigned int xsize = x.shape[0]
    F = np.zeros((ncelx,ncely),dtype=np.double)
    
    for i in range(xsize):
        wtxr = x[i,0]/dx; wtyr = x[i,1]/dy
        indx = <unsigned int> wtxr; indy = <unsigned int> wtyr
        wtxr -= indx; wtyr -= indy
        wtxl = 1.0 - wtxr; wtyl = 1.0 - wtyr
        
        F[indx,indy] += wtxl*wtyl
        
        ## No if-statement method
        indxp1 = (indx+1) % ncelx; indyp1 = (indy+1) % ncely
        F[indxp1,indy] += wtxr*wtyl
        F[indx,indyp1] += wtxl*wtyr
        F[indxp1,indyp1] += wtxr*wtyr
        
    return F/(dx*dy*xsize)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def InterpPar2D(np.ndarray[double, ndim=2] x, np.ndarray[double, ndim=2] F, double dx, double dy, unsigned int ncelx, unsigned int ncely):
    cdef unsigned int i, indx, indxp1, indy, indyp1
    cdef double wtxl, wtxr, wtyl, wtyr, xl, yl
    cdef np.ndarray[np.double_t,ndim=1] v
    cdef unsigned int xsize = x.shape[0]
    v = np.zeros(xsize,dtype=np.double)
    
    for i in range(xsize):
        #indx = <unsigned int>(x[i,0]/dx); indy = <unsigned int>(x[i,1]/dy)
        #xl = indx*dx; yl = indy*dy
        #wtxr = (x[i,0]-xl)/dx; wtxl = 1.0-wtxr
        #wtyr = (x[i,1]-yl)/dy; wtyl = 1.0-wtyl
    
        wtxr = x[i,0]/dx; wtyr = x[i,1]/dy
        indx = <unsigned int> wtxr; indy = <unsigned int> wtyr
        wtxr -= indx; wtyr -= indy
        wtxl = 1.0 - wtxr; wtyl = 1.0 - wtyr
    
        indxp1 = (indx+1) % ncelx; indyp1 = (indy+1) % ncely
        v[i] += wtxl*wtyl*F[indx,indy] + wtxr*wtyl*F[indxp1,indy] + wtxl*wtyr*F[indx,indyp1] + wtxr*wtyr*F[indxp1,indyp1]
            
    return v

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def InterpGrid3D(np.ndarray[double,ndim=2] x, np.ndarray[double,ndim=1] v, double dx, double dy, double dz, unsigned int ncelx, unsigned int ncely, unsigned int ncelz):
    cdef unsigned int i, indx, indy, indz, indxp1, indyp1, indzp1
    cdef double wtxl, wtxr, wtyl, wtyr, wtzl, wtzr, val
    cdef unsigned int xsize = x.shape[0]
    cdef np.ndarray[np.double_t,ndim=3] F
    F = np.zeros((ncelx,ncely,ncelz),dtype=np.double)
    
    for i in range(xsize):
        wtxr = x[i,0]/dx; wtyr = x[i,1]/dy; wtzr = x[i,2]/dz
        indx = <unsigned int> wtxr; indy = <unsigned int> wtyr; indz = <unsigned int> wtzr
        wtxr -= indx; wtyr -= indy; wtzr -= indz
        wtxl = 1.0 - wtxr; wtyl = 1.0 - wtyr; wtzl = 1.0 - wtzr
        
        indxp1 = (indx+1) % ncelx; indyp1 = (indy+1) % ncely; indzp1 = (indz+1) % ncelz

        val = v[i]; wtxl *= val; wtxr *= val

        F[indx,indy,indz] += wtxl*wtyl*wtzl
        F[indxp1,indy,indz] += wtxr*wtyl*wtzl
        F[indx,indyp1,indz] += wtxl*wtyr*wtzl
        F[indx,indy,indzp1] += wtxl*wtyl*wtzr
        F[indxp1,indyp1,indz] += wtxr*wtyr*wtzl
        F[indxp1,indy,indzp1] += wtxr*wtyl*wtzr
        F[indx,indyp1,indzp1] += wtxl*wtyr*wtzr
        F[indxp1,indyp1,indzp1] += wtxr*wtyr*wtzr
        
    return F/(dx*dy*dz*xsize)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def InterpPar3D(np.ndarray[double,ndim=2] x, np.ndarray[double,ndim=3] F, double dx, double dy, double dz, unsigned int ncelx, unsigned int ncely, unsigned int ncelz):
    cdef unsigned int i, indx, indy, indz, indxp1, indyp1, indzp1
    cdef double wtxl, wtxr, wtyl, wtyr, wtzl, wtzr
    cdef unsigned int xsize = x.shape[0]
    cdef np.ndarray[np.double_t,ndim=1] vals
    vals = np.zeros(xsize,dtype=np.double)
    
    for i in range(xsize):
        wtxr = x[i,0]/dx; wtyr = x[i,1]/dy; wtzr = x[i,2]/dz
        indx = <unsigned int> wtxr; indy = <unsigned int> wtyr; indz = <unsigned int> wtzr
        wtxr -= indx; wtyr -= indy; wtzr -= indz
        wtxl = 1.0 - wtxr; wtyl = 1.0 - wtyr; wtzl = 1.0 - wtzr
        
        indxp1 = (indx+1) % ncelx; indyp1 = (indy+1) % ncely; indzp1 = (indz+1) % ncelz

        vals[i] += wtxl*wtyl*wtzl*F[indx,indy,indz] + wtxr*wtyl*wtzl*F[indxp1,indy,indz]
        vals[i] += wtxl*wtyr*wtzl*F[indx,indyp1,indz] + wtxl*wtyl*wtzr*F[indx,indy,indzp1]
        vals[i] += wtxr*wtyr*wtzl*F[indxp1,indyp1,indz] + wtxr*wtyl*wtzr*F[indxp1,indy,indzp1]
        vals[i] += wtxl*wtyr*wtzr*F[indx,indyp1,indzp1] + wtxr*wtyr*wtzr*F[indxp1,indyp1,indzp1]
    
    return vals        