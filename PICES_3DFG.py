# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:45:53 2016

@author: ricketsonl
"""

import numpy as np
#from numpy.linalg import norm
#import pyximport
#pyximport.install(setup_args={"include_dirs":np.get_include()},reload_support=True)
import InterpolationRoutines as IR
import InterpolationRoutines_MV as IRMV
import SpectralOps as SO


class PICES3D:
    chrg = -1.
    B = 0.
    def __init__(self,T_final,domx,domy,domz,numcellsx,numcellsy,numcellsz,numsteps,idata,Bfield_in,Npi):
        self.T = T_final
        self.ncelx = numcellsx; self.ncely = numcellsy; self.ncelz = numcellsz
        self.nstep = numsteps
        self.dt = T_final/numsteps
        self.DomLenX = domx; self.DomLenY = domy; self.DomLenZ = domz
        self.dx = domx/numcellsx; self.dy = domy/numcellsy; self.dz = domz/numcellsz

        self.time = 0.
        
        self.IDatGen = idata
        self.Bfield = Bfield_in

        self.x, self.v = self.IDatGen(Npi,domx,domy,domz)
        if np.amin(self.x) < 0. or np.amax(self.x[:,0]) > self.DomLenX or np.amax(self.x[:,1]) > self.DomLenY or np.amax(self.x[:,2]) > self.DomLenZ:
            print('Warning: Some particles initialized outside domain')

        self.rho = np.zeros((self.ncelx,self.ncely,self.ncelz))
        self.E = np.zeros((self.ncelx,self.ncely,self.ncelz,3))
        
        self.KE = np.zeros(numsteps+1)
        self.PE = np.zeros(numsteps+1)
        
        self.Epar = np.zeros((Npi,3)); self.phi = np.zeros((self.ncelx,self.ncely,self.ncelz))
        
            
    def IG(self,vee): ## Grid interpolation wrapper function
        F = IR.InterpGrid3D(self.x,vee,self.dx,self.dy,self.dz,self.ncelx,self.ncely,self.ncelz)
        return F
        
    def IGVec(self,vee):
        F = np.zeros((self.ncelx,self.ncely,self.ncelz,3))
        F[:,:,:,0] = IR.InterpGrid3D(self.x,vee[:,0],self.dx,self.dy,self.dz,self.ncelx,self.ncely,self.ncelz)
        F[:,:,:,1] = IR.InterpGrid3D(self.x,vee[:,1],self.dx,self.dy,self.dz,self.ncelx,self.ncely,self.ncelz)
        F[:,:,:,2] = IR.InterpGrid3D(self.v,vee[:,2],self.dx,self.dy,self.dz,self.ncelx,self.ncely,self.ncelz)
        return F        
        
    def IPVec(self,EE):
        F = np.zeros_like(self.x)
        F[:,0] = IRMV.InterpPar3D(self.x,EE[:,:,:,0],self.dx,self.dy,self.dz,self.ncelx,self.ncely,self.ncelz)
        F[:,1] = IRMV.InterpPar3D(self.x,EE[:,:,:,1],self.dx,self.dy,self.dz,self.ncelx,self.ncely,self.ncelz)
        F[:,2] = IRMV.InterpPar3D(self.x,EE[:,:,:,2],self.dx,self.dy,self.dz,self.ncelx,self.ncely,self.ncelz)
        return F
        
    def Div(self,F):
        Fx_x, Fx_y, Fx_z = SO.SpectralDerivative3D(F[:,:,0],self.DomLenX,self.DomLenY,self.DomLenZ)
        Fy_x, Fy_y, Fy_z = SO.SpectralDerivative3D(F[:,:,1],self.DomLenX,self.DomLenY,self.DomLenZ)
        Fz_x, Fz_y, Fz_z = SO.SpectralDerivative3D(F[:,:,2],self.DomLenX,self.DomLenY,self.DomLenZ)
        return Fx_x+Fy_y+Fz_z
    
    def PushPars(self,effE):
        self.Epar = self.IPVec(effE)
        self.v += self.dt*self.chrg*self.Epar
        self.x += self.dt*self.v
        self.x[:,0] %= self.DomLenX; self.x[:,1] %= self.DomLenY; self.x[:,2] %= self.DomLenZ
        
    def InterpGridHydros(self,i):
        self.rho = self.IG(np.ones(self.x.shape[0]))
        #self.u[i] = self.IGVec(self.v)
        #self.energy[i] = self.IG(self.v[:,0]**2 + self.v[:,1]**2)
        self.KE[i] = 0.5*np.sum(self.v[:,0]**2 + self.v[:,1]**2 + self.v[:,2]**2)/self.x.shape[0]
        
    def Initialize(self):
        #self.x, self.v = self.IDatGen(N,self.DomLenX,self.DomLenY,self.DomLenZ)
        self.InterpGridHydros(0)
        self.phi = SO.Poisson3Dperiodic(self.chrg*(self.rho - np.mean(self.rho))*self.DomLenX*self.DomLenY*self.DomLenZ,self.DomLenX,self.DomLenY,self.DomLenZ)
        self.E[:,:,:,0], self.E[:,:,:,1], self.E[:,:,:,2] = SO.SpectralDerivative3D(-1.*self.phi,self.DomLenX,self.DomLenY,self.DomLenZ)
        
    def ComputeFieldPoisson(self,i):
        self.phi = SO.Poisson3Dperiodic(self.chrg*(self.rho - np.mean(self.rho))*self.DomLenX*self.DomLenY*self.DomLenZ,self.DomLenX,self.DomLenY,self.DomLenZ)
        self.E[:,:,:,0], self.E[:,:,:,1], self.E[:,:,:,2] = SO.SpectralDerivative3D(-1.*self.phi,self.DomLenX,self.DomLenY,self.DomLenZ)
        
    def Run(self,notify=50):
        self.Initialize()
        for i in range(0,self.nstep):
            self.ComputeFieldPoisson(i)
            self.PE[i] = 0.5*np.sum(self.phi*self.rho)*self.chrg*self.dx*self.dy*self.dz
            self.PushPars(self.E)
            self.InterpGridHydros(i+1)
            if i % notify == 0:
                print('Completed time-step ' + str(i) + ': Simulation time is ' + str(self.time))
            self.time += self.dt
        self.ComputeFieldPoisson(self.nstep)
        self.PE[self.nstep] = 0.5*np.sum(self.phi*self.rho)*self.chrg*self.dx*self.dy*self.dz
        
    def KEnergy(self):
        return 0.5*np.sum(self.energy,axis=(1,2))*self.dx*self.dy
    """def PEnergy(self):
        #PE = self.E[:,:,:,1]**2 + self.E[:,:,:,0]**2
        #return 0.5*np.sum(PE,axis=(1,2))*self.dx*self.dy/(self.DomLenX*self.DomLenY)
        return 0.5*np.sum(self.phi*self.rho,axis=(1,2,3))*self.chrg*self.dx*self.dy*self.dz"""
        
