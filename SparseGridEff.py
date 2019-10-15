# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 12:37:54 2016

@author: ricketsonl
"""

import numpy as np
import SparseGridInterpRoutines as SpIR
import SpectralOps as SO

class SparseGridEff2D:
    def __init__(self,x1,y1,nmax,interp_mode='semicubic',shape_mode='linear'):
        self.pgrids = np.zeros((nmax,2**(nmax+1)), dtype=np.double)
        self.mgrids = np.zeros((nmax-1,2**nmax), dtype=np.double)
        self.Lx = x1; self.Ly = y1
        self.n = nmax
        
        self.dxp = np.array([x1/(2**i) for i in range(1,nmax+1)])
        self.dyp = np.array([y1/(2**i) for i in range(nmax,0,-1)])
        
        self.ncelxp = np.array([2**i for i in range(1,nmax+1)])
        self.ncelyp = np.array([2**i for i in range(nmax,0,-1)])
        
        self.dxm = np.array([x1/(2**i) for i in range(1,nmax)])
        self.dym = np.array([y1/(2**i) for i in range(nmax-1,0,-1)])
        
        self.ncelxm = np.array([2**i for i in range(1,nmax)])
        self.ncelym = np.array([2**i for i in range(nmax-1,0,-1)])
        
        self.imode = interp_mode; self.smode = shape_mode
        
    def InterpolateData(self,x,vee):
        if self.smode == 'linear':
            SpIR.InterpSparseGrid2DOther(x,vee,self.pgrids,self.mgrids,self.dxp,self.dyp)
        else:
            SpIR.InterpSparseGrid2Dbicubic(x,vee,self.pgrids,self.mgrids,self.dxp,self.dyp)
        
    def InterpolateDataBicubic(self,x,vee):
        SpIR.InterpSparseGrid2Dbicubic(x,vee,self.pgrids,self.mgrids,self.dxp,self.dyp)
        
    def InputFunc(self,f):
        for i in range(self.n):
            x = np.linspace(0.,self.Lx,self.ncelxp[i],endpoint=False)
            y = np.linspace(0.,self.Ly,self.ncelyp[i],endpoint=False)
            Y,X = np.meshgrid(x,y)
            p = f(X,Y)
            self.pgrids[i] = p.reshape(2**(self.n+1))
        for i in range(self.n-1):
            x = np.linspace(0.,self.Lx,self.ncelxm[i],endpoint=False)
            y = np.linspace(0.,self.Ly,self.ncelym[i],endpoint=False)
            Y,X = np.meshgrid(x,y)
            m = f(X,Y)
            self.mgrids[i] = m.reshape(2**(self.n))

    ## Zero out all but pgrids[j] so that we essentially just have a regular grid at 
    ## that resolution.  Useful primarily for debugging.
    def ZeroOutAllButOne(self,j):
        for i in range(self.n):
            if not i == j:
                self.pgrids[i] = 0.
        for i in range(self.n-1):
            self.mgrids[i] = 0.
        
    def EvaluateAt(self,x):
        if self.imode == 'cubic':
            return SpIR.EvaluateAtBicubic(x,self.pgrids,self.mgrids,self.dxp,self.dyp)
        else:
            return SpIR.EvaluateAt(x,self.pgrids,self.mgrids,self.dxp,self.dyp)
        
    def EvaluateAtBicubic(self,x):
        return SpIR.EvaluateAtBicubic(x,self.pgrids,self.mgrids,self.dxp,self.dyp)
        
    def RegularGrid(self,nx,ny):
        x = np.linspace(0.,self.Lx,nx,endpoint=False)
        y = np.linspace(0.,self.Ly,ny,endpoint=False)
        if self.imode == 'linear':
            return SpIR.EvaluateOnGrid2D(x,y,self.pgrids,self.mgrids,self.dxp,self.dyp)
        else:
            return SpIR.EvaluateOnGrid2Dbicubic(x,y,self.pgrids,self.mgrids,self.dxp,self.dyp)
        
    def RegularGridBicubic(self,nx,ny):
        x = np.linspace(0.,self.Lx,nx,endpoint=False)
        y = np.linspace(0.,self.Ly,ny,endpoint=False)
        return SpIR.EvaluateOnGrid2Dbicubic(x,y,self.pgrids,self.mgrids,self.dxp,self.dyp)
    
    def SparsePoisson(self):
        u = SparseGridEff2D(self.Lx,self.Ly,self.n,interp_mode=self.imode,shape_mode=self.smode)
        for i in range(self.n):
            p = self.RegularGrid(self.ncelxp[i],self.ncelyp[i])
            up = SO.Poisson2Dperiodic(p,self.Lx,self.Ly)
            u.pgrids[i] = up.transpose().reshape(2**(self.n+1))
        for i in range(self.n-1):
            p = self.RegularGrid(self.ncelxm[i],self.ncelym[i])
            up = SO.Poisson2Dperiodic(p,self.Lx,self.Ly)
            u.mgrids[i] = up.transpose().reshape(2**self.n)
        return u
    
    def SparseDerivative(self):
        fx = SparseGridEff2D(self.Lx,self.Ly,self.n,interp_mode=self.imode,shape_mode=self.smode)
        fy = SparseGridEff2D(self.Lx,self.Ly,self.n,interp_mode=self.imode,shape_mode=self.smode)
        for i in range(self.n):
            p = self.RegularGrid(self.ncelxp[i],self.ncelyp[i])
            fyp, fxp = SO.SpectralDerivative2D(p,self.Lx,self.Ly)
            fx.pgrids[i] = fxp.transpose().reshape(2**(self.n+1)); fy.pgrids[i] = fyp.transpose().reshape(2**(self.n+1))
        for i in range(self.n-1):
            p = self.RegularGrid(self.ncelxm[i],self.ncelym[i])
            fyp, fxp = SO.SpectralDerivative2D(p,self.Lx,self.Ly)
            fx.mgrids[i] = fxp.transpose().reshape(2**(self.n)); fy.mgrids[i] = fyp.transpose().reshape(2**(self.n))
        return fx, fy
        
    def Mean(self):
        return np.sum(self.pgrids)/self.pgrids.shape[1] - np.sum(self.mgrids)/self.mgrids.shape[1]
        
    def Integrate(self):
        integral = 0.
        for i in range(self.n):
            integral += np.sum(self.pgrids[i])*self.dxp[i]*self.dyp[i]
        for i in range(self.n-1):
            integral -= np.sum(self.mgrids[i])*self.dxm[i]*self.dym[i]
        return integral        
        
    ################################################################
    ## Overloading arithmetic operators to do the expected things ##
    ################################################################
    def __mul__(self,fac):
        result = SparseGridEff2D(self.Lx,self.Ly,self.n,interp_mode=self.imode,shape_mode=self.smode)
        if isinstance(fac,(int,float,np.double,np.float,np.int)) :       
            result.pgrids = fac*self.pgrids; result.mgrids = fac*self.mgrids
            return result
        elif isinstance(fac,SparseGridEff2D):
            result.pgrids = fac.pgrids*self.pgrids; result.mgrids = fac.mgrids*self.mgrids
            return result
        else:
            return NotImplemented
        
    def __rmul__(self,fac):
        result = SparseGridEff2D(self.Lx,self.Ly,self.n,interp_mode=self.imode,shape_mode=self.smode)
        if isinstance(fac,(int,float,np.double,np.float,np.int)) :       
            result.pgrids = fac*self.pgrids; result.mgrids = fac*self.mgrids
            return result
        elif isinstance(fac,SparseGridEff2D):
            result.pgrids = fac.pgrids*self.pgrids; result.mgrids = fac.mgrids*self.mgrids
            return result
        else:
            return NotImplemented
        
    def __add__(self,addend):
        result = SparseGridEff2D(self.Lx,self.Ly,self.n,interp_mode=self.imode,shape_mode=self.smode)
        if isinstance(addend,SparseGridEff2D):        
            result.pgrids = self.pgrids + addend.pgrids; result.mgrids = self.mgrids + addend.mgrids
            return result
        elif isinstance(addend,(int,float,np.double,np.float,np.int)):
            result.pgrids = self.pgrids + addend; result.mgrids = self.mgrids + addend
            return result
        else:
            return NotImplemented
            
    def __radd__(self,addend):
        result = SparseGridEff2D(self.Lx,self.Ly,self.n,interp_mode=self.imode,shape_mode=self.smode)
        if isinstance(addend,SparseGridEff2D):        
            result.pgrids = self.pgrids + addend.pgrids; result.mgrids = self.mgrids + addend.mgrids
            return result
        elif isinstance(addend,(int,float,np.double,np.float,np.int)):
            result.pgrids = self.pgrids + addend; result.mgrids = self.mgrids + addend
            return result
        else:
            return NotImplemented
            
    def __sub__(self,addend):
        result = SparseGridEff2D(self.Lx,self.Ly,self.n,interp_mode=self.imode,shape_mode=self.smode)
        if isinstance(addend,SparseGridEff2D):        
            result.pgrids = self.pgrids - addend.pgrids; result.mgrids = self.mgrids - addend.mgrids
            return result
        elif isinstance(addend,(int,float,np.double,np.float,np.int)):
            result.pgrids = self.pgrids - addend; result.mgrids = self.mgrids - addend
            return result
        else:
            return NotImplemented
            
    def __rsub__(self,addend):
        result = SparseGridEff2D(self.Lx,self.Ly,self.n,interp_mode=self.imode,shape_mode=self.smode)
        if isinstance(addend,SparseGridEff2D):        
            result.pgrids = self.pgrids - addend.pgrids; result.mgrids = self.mgrids - addend.mgrids
            return result
        elif isinstance(addend,(int,float,np.double,np.float,np.int)):
            result.pgrids = self.pgrids - addend; result.mgrids = self.mgrids - addend
            return result
        else:
            return NotImplemented
            
## Non-periodic 2-D sparse grid (free space) ###################################################

class SparseGridEff2D_FS:
    def __init__(self,x1,y1,nmax):
        self.pgrids = np.zeros((nmax,2**(nmax+1)), dtype=np.double)
        self.mgrids = np.zeros((nmax-1,2**nmax), dtype=np.double)
        self.Lx = x1; self.Ly = y1
        self.n = nmax
        
        self.dxp = np.array([x1/(2**i) for i in range(1,nmax+1)])
        self.dyp = np.array([y1/(2**i) for i in range(nmax,0,-1)])
        
        self.dxm = np.array([x1/(2**i) for i in range(1,nmax)])
        self.dym = np.array([y1/(2**i) for i in range(nmax-1,0,-1)])
        
    def InterpolateData(self,x,vee):
        SpIR.InterpSparseGrid2DOther(x,vee,self.pgrids,self.mgrids,self.dxp,self.dyp)
        
    def EvaluateAt(self,x):
        return SpIR.EvaluateAt(x,self.pgrids,self.mgrids,self.dxp,self.dyp)
        
    def RegularGrid(self,nx,ny):
        g = np.zeros((nx,ny))
        x = np.linspace(0.,self.Lx,nx,endpoint=False)
        y = np.linspace(0.,self.Ly,ny,endpoint=False)
        
        for i in range(y.shape[0]):
            X = np.column_stack((x,y[i]*np.ones(x.shape[0])))
            g[:,i] = self.EvaluateAt(X)
        
        return g
    
    def SparseFieldSolveFreeSpace(self):
        ux = SparseGridEff2D(self.Lx,self.Ly,self.n)
        uy = SparseGridEff2D(self.Lx,self.Ly,self.n)
        for i in range(self.n):
            p = self.pgrids[i].reshape((2**(self.n-i),2**(i+1))).transpose()
            hx = self.dxp[i]; hy = self.dyp[i]
            upx, upy = fmain.solve.solve_poisson(hx,hy,p)
            ux.pgrids[i] = -np.real(upx.transpose().reshape(2**(self.n+1)))
            uy.pgrids[i] = -np.real(upy.transpose().reshape(2**(self.n+1)))
        for i in range(self.n-1):
            p = self.mgrids[i].reshape((2**(self.n-i-1),2**(i+1))).transpose()
            hx = self.dxm[i]; hy = self.dym[i]
            upx, upy = fmain.solve.solve_poisson(hx,hy,p)
            ux.mgrids[i] = -np.real(upx.transpose().reshape(2**(self.n)))
            uy.mgrids[i] = -np.real(upy.transpose().reshape(2**(self.n)))
        return ux, uy
        
    
    def SparsePoisson(self):
        u = SparseGridEff2D(self.Lx,self.Ly,self.n)
        for i in range(self.n):
            p = self.pgrids[i].reshape((2**(self.n-i),2**(i+1))).transpose()
            up = SO.Poisson2Dperiodic(p,self.Lx,self.Ly)
            u.pgrids[i] = up.transpose().reshape(2**(self.n+1))
        for i in range(self.n-1):
            p = self.mgrids[i].reshape((2**(self.n-i-1),2**(i+1))).transpose()
            up = SO.Poisson2Dperiodic(p,self.Lx,self.Ly)
            u.mgrids[i] = up.transpose().reshape(2**self.n)
        return u
        
    def SparseDerivative(self):
        fx = SparseGridEff2D(self.Lx,self.Ly,self.n)
        fy = SparseGridEff2D(self.Lx,self.Ly,self.n)
        for i in range(self.n):
            p = self.pgrids[i].reshape((2**(self.n-i),2**(i+1))).transpose()
            fyp, fxp = SO.SpectralDerivative2D(p,self.Lx,self.Ly)
            fx.pgrids[i] = fxp.transpose().reshape(2**(self.n+1)); fy.pgrids[i] = fyp.transpose().reshape(2**(self.n+1))
        for i in range(self.n-1):
            p = self.mgrids[i].reshape((2**(self.n-i-1),2**(i+1))).transpose()
            fyp, fxp = SO.SpectralDerivative2D(p,self.Lx,self.Ly)
            fx.mgrids[i] = fxp.transpose().reshape(2**(self.n)); fy.mgrids[i] = fyp.transpose().reshape(2**(self.n))
        return fx, fy
        
    def Mean(self):
        return np.sum(self.pgrids)/self.pgrids.shape[1] - np.sum(self.mgrids)/self.mgrids.shape[1]
        
    def Integrate(self):
        integral = 0.
        for i in range(self.n):
            integral += np.sum(self.pgrids[i])*self.dxp[i]*self.dyp[i]
        for i in range(self.n-1):
            integral -= np.sum(self.mgrids[i])*self.dxm[i]*self.dym[i]
        return integral
        
        
    ################################################################
    ## Overloading arithmetic operators to do the expected things ##
    ################################################################
    def __mul__(self,fac):
        result = SparseGridEff2D(self.Lx,self.Ly,self.n)
        if isinstance(fac,(int,float,np.double,np.float,np.int)) :       
            result.pgrids = fac*self.pgrids; result.mgrids = fac*self.mgrids
            return result
        elif isinstance(fac,SparseGridEff2D):
            result.pgrids = fac.pgrids*self.pgrids; result.mgrids = fac.mgrids*self.mgrids
            return result
        else:
            return NotImplemented
        
    def __rmul__(self,fac):
        result = SparseGridEff2D(self.Lx,self.Ly,self.n)
        if isinstance(fac,(int,float,np.double,np.float,np.int)) :       
            result.pgrids = fac*self.pgrids; result.mgrids = fac*self.mgrids
            return result
        elif isinstance(fac,SparseGridEff2D):
            result.pgrids = fac.pgrids*self.pgrids; result.mgrids = fac.mgrids*self.mgrids
            return result
        else:
            return NotImplemented
        
    def __add__(self,addend):
        result = SparseGridEff2D(self.Lx,self.Ly,self.n)
        if isinstance(addend,SparseGridEff2D):        
            result.pgrids = self.pgrids + addend.pgrids; result.mgrids = self.mgrids + addend.mgrids
            return result
        elif isinstance(addend,(int,float,np.double,np.float,np.int)):
            result.pgrids = self.pgrids + addend; result.mgrids = self.mgrids + addend
            return result
        else:
            return NotImplemented
            
    def __radd__(self,addend):
        result = SparseGridEff2D(self.Lx,self.Ly,self.n)
        if isinstance(addend,SparseGridEff2D):        
            result.pgrids = self.pgrids + addend.pgrids; result.mgrids = self.mgrids + addend.mgrids
            return result
        elif isinstance(addend,(int,float,np.double,np.float,np.int)):
            result.pgrids = self.pgrids + addend; result.mgrids = self.mgrids + addend
            return result
        else:
            return NotImplemented
            
    def __sub__(self,addend):
        result = SparseGridEff2D(self.Lx,self.Ly,self.n)
        if isinstance(addend,SparseGridEff2D):        
            result.pgrids = self.pgrids - addend.pgrids; result.mgrids = self.mgrids - addend.mgrids
            return result
        elif isinstance(addend,(int,float,np.double,np.float,np.int)):
            result.pgrids = self.pgrids - addend; result.mgrids = self.mgrids - addend
            return result
        else:
            return NotImplemented
            
    def __rsub__(self,addend):
        result = SparseGridEff2D(self.Lx,self.Ly,self.n)
        if isinstance(addend,SparseGridEff2D):        
            result.pgrids = self.pgrids - addend.pgrids; result.mgrids = self.mgrids - addend.mgrids
            return result
        elif isinstance(addend,(int,float,np.double,np.float,np.int)):
            result.pgrids = self.pgrids - addend; result.mgrids = self.mgrids - addend
            return result
        else:
            return NotImplemented

## 3-D Sparse Grid Class ##########################################################
            
class SparseGridEff3D:
    def __init__(self,x1,y1,z1,nmax,interp_mode='semicubic',shape_mode='linear'):
        self.n = nmax
        self.Lengths = np.asarray([x1,y1,z1],dtype=np.double)
        
        self.num_np2grids = int((nmax+1)*(nmax)/2)
        self.num_np1grids = int((nmax-1)*nmax/2)
        self.num_ngrids = int((nmax-2)*(nmax-1)/2)
        
        self.np2grids = np.zeros((self.num_np2grids,2**(nmax+2)), dtype=np.double)
        self.np1grids = np.zeros((self.num_np1grids,2**(nmax+1)), dtype=np.double)
        self.ngrids = np.zeros((self.num_ngrids,2**nmax), dtype=np.double)
        
        self.ncnp2 = np.zeros((3,self.num_np2grids),dtype=np.dtype("i"))
        self.ncnp1 = np.zeros((3,self.num_np1grids),dtype=np.dtype("i"))
        self.ncn = np.zeros((3,self.num_ngrids),dtype=np.dtype("i"))
        
        self.imode = interp_mode; self.smode = shape_mode
        
        ct = 0
        for i in range(1,nmax+1):
            for j in range(1,nmax+2-i):
                self.ncnp2[0,ct] = 2**i; self.ncnp2[1,ct] = 2**j; self.ncnp2[2,ct] = 2**(nmax+2-i-j)                
                ct += 1            
                
        ct = 0
        for i in range(1,nmax):
            for j in range(1,nmax+1-i):
                self.ncnp1[0,ct] = 2**i; self.ncnp1[1,ct] = 2**j; self.ncnp1[2,ct] = 2**(nmax+1-i-j)                
                ct += 1
        
        ct = 0
        for i in range(1,nmax-1):
            for j in range(1,nmax-i):
                self.ncn[0,ct] = 2**i; self.ncn[1,ct] = 2**j; self.ncn[2,ct] = 2**(nmax-i-j)                
                ct += 1
                
        
    def InterpolateData(self,x,v):
        SpIR.InterpSparseGrid3D(x,v,self.np2grids,self.np1grids,self.ngrids,self.ncnp2,self.ncnp1,self.ncn,self.Lengths)
        
#    def InputFunc(self,F):
#        for i in range(self.num_np2grids):        
#            x = np.linspace(0.,self.Lengths[0],self.ncnp2[0,i],endpoint=False)
#            y = np.linspace(0.,self.Lengths[0],self.ncnp2[1,i],endpoint=False)
#            z = np.linspace(0.,self.Lengths[0],self.ncnp2[2,i],endpoint=False)
#            X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
#            p = F(X,Y,Z)
#            self.np2grids[i] = p.reshape()
        
    def EvaluateAt(self,x):
        return SpIR.EvaluateAt3D(x,self.np2grids,self.np1grids,self.ngrids,self.ncnp2,self.ncnp1,self.ncn,self.Lengths)
        
    def RegularGrid(self,nx,ny,nz):
        g = np.zeros((nx,ny,nz))
        x = np.linspace(0.,self.Lengths[0],nx,endpoint=False)
        y = np.linspace(0.,self.Lengths[1],ny,endpoint=False)
        z = np.linspace(0.,self.Lengths[2],nz,endpoint=False)
        #for i in range(y.shape[0]):
        #    for j in range(z.shape[0]):
        #        X = np.column_stack((x,y[i]*np.ones(x.shape[0]),z[j]*np.ones(x.shape[0])))
        #        g[:,i,j] = self.EvaluateAt(X)
        if self.imode == 'linear':
            g = SpIR.EvaluateOnGrid3D(x,y,z,self.np2grids,self.np1grids,self.ngrids,self.ncnp2,self.ncnp1,self.ncn,self.Lengths)
        else:
            g = SpIR.EvaluateOnGrid3Dbicubic(x,y,z,self.np2grids,self.np1grids,self.ngrids,self.ncnp2,self.ncnp1,self.ncn,self.Lengths)
        
        return g
        
    def Regular2DSlice(self,direction,val,n1,n2):
        g = np.zeros((n1,n2))
        if direction == 'z':
            x = np.linspace(0.,self.Lengths[0],n1,endpoint=False)
            y = np.linspace(0.,self.Lengths[1],n2,endpoint=False)
            for i in range(y.shape[0]):
                X = np.column_stack((x,y[i]*np.ones(x.shape[0]),val*np.ones(x.shape[0])))
                g[:,i] = self.EvaluateAt(X)
            return g
        elif direction == 'y':
            x = np.linspace(0.,self.Lengths[0],n1,endpoint=False)
            z = np.linspace(0.,self.Lengths[2],n2,endpoint=False)
            for i in range(z.shape[0]):
                X = np.column_stack((x,val*np.ones(x.shape[0]),z[i]*np.ones(x.shape[0])))
                g[:,i] = self.EvaluateAt(X)
            return g
        elif direction == 'x':
            y = np.linspace(0.,self.Lengths[1],n1,endpoint=False)
            z = np.linspace(0.,self.Lengths[2],n2,endpoint=False)
            for i in range(z.shape[0]):
                X = np.column_stack((val*np.ones(y.shape[0]),y,z[i]*np.ones(y.shape[0])))
                g[:,i] = self.EvaluateAt(X)
            return g
        else:
            print('Invalid direction argument')
            return g
            
    def SparsePoisson(self):
        u = SparseGridEff3D(self.Lengths[0],self.Lengths[1],self.Lengths[2],self.n,interp_mode=self.imode,shape_mode=self.smode)
        
        ct = 0
        for i in range(1,self.n+1):
            for j in range(1,self.n+2-i):
                #p = self.np2grids[ct].reshape((2**(self.n+2-i-j),2**j,2**i)).transpose()
                p = self.RegularGrid(2**i, 2**j, 2**(self.n+2-i-j))
                up = SO.Poisson3Dperiodic(p,self.Lengths[0],self.Lengths[1],self.Lengths[2])
                u.np2grids[ct] = up.transpose().reshape(2**(self.n+2))                
                ct += 1 
        
        ct = 0
        for i in range(1,self.n):
            for j in range(1,self.n+1-i):
                #p = self.np1grids[ct].reshape((2**(self.n+1-i-j),2**j,2**i)).transpose()
                p = self.RegularGrid(2**i,2**j,2**(self.n+1-i-j))
                up = SO.Poisson3Dperiodic(p,self.Lengths[0],self.Lengths[1],self.Lengths[2])
                u.np1grids[ct] = up.transpose().reshape(2**(self.n+1))                
                ct += 1
        
        ct = 0
        for i in range(1,self.n-1):
            for j in range(1,self.n-i):
                #p = self.ngrids[ct].reshape((2**(self.n-i-j),2**j,2**i)).transpose()
                p = self.RegularGrid(2**i,2**j,2**(self.n-i-j))
                up = SO.Poisson3Dperiodic(p,self.Lengths[0],self.Lengths[1],self.Lengths[2])
                u.ngrids[ct] = up.transpose().reshape(2**(self.n))                
                ct += 1
        return u
        
    def SparseDerivative(self):
        fx = SparseGridEff3D(self.Lengths[0],self.Lengths[1],self.Lengths[2],self.n,interp_mode=self.imode,shape_mode=self.smode)
        fy = SparseGridEff3D(self.Lengths[0],self.Lengths[1],self.Lengths[2],self.n,interp_mode=self.imode,shape_mode=self.smode)
        fz = SparseGridEff3D(self.Lengths[0],self.Lengths[1],self.Lengths[2],self.n,interp_mode=self.imode,shape_mode=self.smode)
        
        ct = 0
        for i in range(1,self.n+1):
            for j in range(1,self.n+2-i):
                #p = self.np2grids[ct].reshape((2**(self.n+2-i-j),2**j,2**i)).transpose()
                p = self.RegularGrid(2**i, 2**j, 2**(self.n+2-i-j))
                fxp, fyp, fzp = SO.SpectralDerivative3D(p,self.Lengths[0],self.Lengths[1],self.Lengths[2])
                fx.np2grids[ct] = fxp.transpose().reshape(2**(self.n+2))  
                fy.np2grids[ct] = fyp.transpose().reshape(2**(self.n+2))
                fz.np2grids[ct] = fzp.transpose().reshape(2**(self.n+2))
                ct += 1 
        
        ct = 0
        for i in range(1,self.n):
            for j in range(1,self.n+1-i):
                #p = self.np1grids[ct].reshape((2**(self.n+1-i-j),2**j,2**i)).transpose()
                p = self.RegularGrid(2**i,2**j,2**(self.n+1-i-j))
                fxp, fyp, fzp = SO.SpectralDerivative3D(p,self.Lengths[0],self.Lengths[1],self.Lengths[2])
                fx.np1grids[ct] = fxp.transpose().reshape(2**(self.n+1))   
                fy.np1grids[ct] = fyp.transpose().reshape(2**(self.n+1))  
                fz.np1grids[ct] = fzp.transpose().reshape(2**(self.n+1))  
                ct += 1
        
        ct = 0
        for i in range(1,self.n-1):
            for j in range(1,self.n-i):
                #p = self.ngrids[ct].reshape((2**(self.n-i-j),2**j,2**i)).transpose()
                p = self.RegularGrid(2**i,2**j,2**(self.n-i-j))
                fxp, fyp, fzp = SO.SpectralDerivative3D(p,self.Lengths[0],self.Lengths[1],self.Lengths[2])
                fx.ngrids[ct] = fxp.transpose().reshape(2**(self.n))    
                fy.ngrids[ct] = fyp.transpose().reshape(2**(self.n))  
                fz.ngrids[ct] = fzp.transpose().reshape(2**(self.n))  
                ct += 1

        return fx, fy, fz
        
    def Mean(self):
        return np.sum(self.np2grids)/self.np2grids.shape[1] - 2.*np.sum(self.np1grids)/self.np1grids.shape[1] + np.sum(self.ngrids)/self.ngrids.shape[1]

    def Integrate(self):
        integral = 0.
        for i in range(self.num_np2grids):
            integral += np.sum(self.np2grids[i])*self.Lengths[0]*self.Lengths[1]*self.Lengths[2]/(self.ncnp2[0,i]*self.ncnp2[1,i]*self.ncnp2[2,i])
        for i in range(self.num_np1grids):
            integral -= 2.*np.sum(self.np1grids[i])*self.Lengths[0]*self.Lengths[1]*self.Lengths[2]/(self.ncnp1[0,i]*self.ncnp1[1,i]*self.ncnp1[2,i])
        for i in range(self.num_ngrids):
            integral += np.sum(self.ngrids[i])*self.Lengths[0]*self.Lengths[1]*self.Lengths[2]/(self.ncn[0,i]*self.ncn[1,i]*self.ncn[2,i])
            
        return integral
    
    ################################################################
    ## Overloading arithmetic operators to do the expected things ##
    ################################################################
    def __mul__(self,fac):
        result = SparseGridEff3D(self.Lengths[0],self.Lengths[1],self.Lengths[2],self.n,interp_mode=self.imode,shape_mode=self.smode)
        if isinstance(fac,(int,float,np.double,np.float,np.int)) :       
            result.np2grids = fac*self.np2grids; result.np1grids = fac*self.np1grids; result.ngrids = fac*self.ngrids
            return result
        elif isinstance(fac,SparseGridEff3D):
            result.np2grids = fac.np2grids*self.np2grids; result.np1grids = fac.np1grids*self.np1grids
            result.ngrids = fac.ngrids*self.ngrids
            return result
        else:
            return NotImplemented
        
    def __rmul__(self,fac):
        result = SparseGridEff3D(self.Lengths[0],self.Lengths[1],self.Lengths[2],self.n,interp_mode=self.imode,shape_mode=self.smode)
        if isinstance(fac,(int,float,np.double,np.float,np.int)) :       
            result.np2grids = fac*self.np2grids; result.np1grids = fac*self.np1grids; result.ngrids = fac*self.ngrids
            return result
        elif isinstance(fac,SparseGridEff3D):
            result.np1grids = fac.np2grids*self.np2grids; result.np1grids = fac.np1grids*self.np1grids
            result.ngrids = fac.ngrids*self.ngrids
            return result
        else:
            return NotImplemented
        
    def __add__(self,addend):
        result = SparseGridEff3D(self.Lengths[0],self.Lengths[1],self.Lengths[2],self.n,interp_mode=self.imode,shape_mode=self.smode)
        if isinstance(addend,SparseGridEff3D):        
            result.np2grids = self.np2grids + addend.np2grids; result.np1grids = self.np1grids + addend.np1grids
            result.ngrids = self.ngrids + addend.ngrids
            return result
        elif isinstance(addend,(int,float,np.double,np.float,np.int)):
            result.np2grids = self.np2grids + addend; result.np1grids = self.np1grids + addend
            result.ngrids = self.ngrids + addend
            return result
        else:
            return NotImplemented
            
    def __radd__(self,addend):
        result = SparseGridEff3D(self.Lengths[0],self.Lengths[1],self.Lengths[2],self.n,interp_mode=self.imode,shape_mode=self.smode)
        if isinstance(addend,SparseGridEff2D):        
            result.np2grids = self.np2grids + addend.np2grids; result.np1grids = self.np1grids + addend.np1grids
            result.ngrids = self.ngrids + addend.ngrids
            return result
        elif isinstance(addend,(int,float,np.double,np.float,np.int)):
            result.np2grids = self.np2grids + addend; result.np1grids = self.np1grids + addend
            result.ngrids = self.ngrids + addend
            return result
        else:
            return NotImplemented
            
    def __sub__(self,addend):
        result = SparseGridEff3D(self.Lengths[0],self.Lengths[1],self.Lengths[2],self.n,interp_mode=self.imode,shape_mode=self.smode)
        if isinstance(addend,SparseGridEff2D):        
            result.np2grids = self.np2grids - addend.np2grids; result.np1grids = self.np1grids - addend.np1grids
            result.ngrids = self.ngrids - addend.ngrids
            return result
        elif isinstance(addend,(int,float,np.double,np.float,np.int)):
            result.np2grids = self.np2grids - addend; result.np1grids = self.np1grids - addend
            result.ngrids = self.ngrids - addend
            return result
        else:
            return NotImplemented
            
    def __rsub__(self,addend):
        result = SparseGridEff3D(self.Lengths[0],self.Lengths[1],self.Lengths[2],self.n,interp_mode=self.imode,shape_mode=self.smode)
        if isinstance(addend,SparseGridEff2D):        
            result.np2grids = self.np2grids - addend.np2grids; result.np1grids = self.np1grids - addend.np1grids
            result.ngrids = self.ngrids - addend.ngrids
            return result
        elif isinstance(addend,(int,float,np.double,np.float,np.int)):
            result.np2grids = self.np2grids - addend; result.np1grids = self.np1grids - addend
            result.ngrids = self.ngrids - addend
            return result
        else:
            return NotImplemented
