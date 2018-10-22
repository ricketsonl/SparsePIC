# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:44:16 2016

@author: ricketsonl
"""

import numpy as np
#from numpy.linalg import norm
import InterpolationRoutines_MV as IR
import SpectralOps as SO
import SparseGridEff as SG
#from scipy.optimize import newton_krylov
import matplotlib.pyplot as plt
from matplotlib import animation
#plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'

#import Filters as FIL


class PICES2DSG:
    chrg = -1.
    space_norm = 1; time_norm = np.inf # Sets the space- and time-norms for variances
    imptol = 1.e-10 # Tolerance for Newton-Krylov iteration
    B = 10.
    def __init__(self,T_final,domx,domy,numcellsx,numsteps,idata,Npi,field_solve='fast',intmode='semicubic',shmode='linear',ncxf=128,ncyf=128):
        self.T = T_final
        self.n = int(np.log2(numcellsx-0.01))+1
        self.ncelx = 2**self.n; self.ncely = 2**self.n
        self.nstep = numsteps
        self.dt = T_final/numsteps
        self.DomLenX = domx; self.DomLenY = domy
        self.dx = domx/self.ncelx; self.dy = domy/self.ncely
        
        self.IDatGen = idata
        self.time = 0.

        self.x, self.v = self.IDatGen(Npi,domx,domy)
        if np.amin(self.x) < 0. or np.amax(self.x[:,0]) > self.DomLenX or np.amax(self.x[:,1]) > self.DomLenY:
            print('Warning: Some particles initialized outside domain')
            
        self.parwts = np.ones(Npi)
        
        self.field_solve = field_solve
        #self.rho = np.zeros((numsteps+1,self.ncelx,self.ncely)); self.u = np.zeros((numsteps+1,self.ncelx,self.ncely,2))
        #self.energy = np.zeros((numsteps+1,self.ncelx,self.ncely)); self.E = np.zeros((numsteps+1,self.ncelx,self.ncely,2))
        vSparseGrid = np.vectorize(SG.SparseGridEff2D)
        self.rho = vSparseGrid(self.DomLenX*np.ones(numsteps+1),self.DomLenY*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int),interp_mode=intmode,shape_mode=shmode)    
        #self.ux = vSparseGrid(self.DomLenX*np.ones(numsteps+1),self.DomLenY*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int))
        #self.uy = vSparseGrid(self.DomLenX*np.ones(numsteps+1),self.DomLenY*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int))
        #self.energy = vSparseGrid(self.DomLenX*np.ones(numsteps+1),self.DomLenY*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int))
        if self.field_solve == 'fast':      
            self.Ex = vSparseGrid(self.DomLenX*np.ones(numsteps+1),self.DomLenY*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int),interp_mode=intmode,shape_mode=shmode)
            self.Ey = vSparseGrid(self.DomLenX*np.ones(numsteps+1),self.DomLenY*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int),interp_mode=intmode,shape_mode=shmode)
            self.phi = vSparseGrid(self.DomLenX*np.ones(numsteps+1),self.DomLenY*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int),interp_mode=intmode,shape_mode=shmode)
        else:
            self.Ex = np.zeros((numsteps+1,ncxf,ncxf))
            self.Ey = np.zeros((numsteps+1,ncxf,ncxf))
            self.phi = np.zeros((numsteps+1,ncxf,ncxf))
            self.ncxf = ncxf; self.ncyf = ncyf
            self.dxf = domx/ncxf; self.dyf = domy/ncyf
        
        self.KE = np.zeros(numsteps+1)
        
        self.Epar = np.zeros((Npi,2))
    
    def PushPars(self,i):
        if self.field_solve == 'fast':
            self.Epar[:,0] = self.Ex[i].EvaluateAt(self.x); self.Epar[:,1] = self.Ey[i].EvaluateAt(self.x)
        else:
            self.Epar[:,0] = IR.InterpPar2D(self.x,self.Ex[i],self.dxf,self.dyf,self.ncxf,self.ncyf)
            self.Epar[:,1] = IR.InterpPar2D(self.x,self.Ey[i],self.dxf,self.dyf,self.ncxf,self.ncyf)
            
        self.v += 0.5*self.dt*self.chrg*self.Epar
        t = self.chrg*self.B*self.dt/2.0; s = 2.0*t/(1.0 + t*t); c = (1-t*t)/(1+t*t)
        vnewx = self.v[:,0]*c - self.v[:,1]*s
        vnewy = self.v[:,0]*s + self.v[:,1]*c
        self.v[:,0] = vnewx; self.v[:,1] = vnewy
        self.v += 0.5*self.dt*self.chrg*self.Epar
        self.x += self.dt*self.v
        self.x[:,0] %= self.DomLenX; self.x[:,1] %= self.DomLenY
        
    def InterpGridHydros(self,i):
        self.rho[i].InterpolateData(self.x,self.parwts)
        #self.ux[i].InterpolateData(self.x,self.v[:,0])
        #self.uy[i].InterpolateData(self.x,self.v[:,1])
        #self.energy[i].InterpolateData(self.x,self.v[:,0]**2 + self.v[:,1]**2)
        self.KE[i] = 0.5*np.sum(self.v[:,0]*self.v[:,0] + self.v[:,1]*self.v[:,1])/self.x.shape[0]
        
    def Initialize(self):
        self.InterpGridHydros(0)
            
    def ComputeFieldPoisson(self,i):
        if self.field_solve == 'fast':
            PoissRHS = self.chrg*(self.rho[i]-self.rho[i].Mean())*self.DomLenX*self.DomLenY
            self.phi[i] = PoissRHS.SparsePoisson()
            self.Ex[i], self.Ey[i] = self.phi[i].SparseDerivative()
            self.Ex[i] = -1.*self.Ex[i]; self.Ey[i] = -1.*self.Ey[i]
        else:
            rhoreg = self.rho[i].RegularGrid(self.ncxf,self.ncyf)
            self.phi[i] = SO.Poisson2Dperiodic(self.chrg*(rhoreg-np.mean(rhoreg))*self.DomLenX*self.DomLenY,self.DomLenX,self.DomLenY)
            self.Ey[i], self.Ex[i] = SO.SpectralDerivative2D(-1.*self.phi[i],self.DomLenX,self.DomLenY)            
            
    def Run(self,notify=50):
        self.Initialize()
        for i in range(0,self.nstep):
            self.ComputeFieldPoisson(i)
            self.PushPars(i)
            self.InterpGridHydros(i+1)
            if i % notify == 0:
                print('Completed time-step '  + str(i) + ': Simulation time is ' + str(self.time))
            self.time += self.dt
        self.ComputeFieldPoisson(self.nstep)
        
    def KEnergy(self):
        KE = np.zeros(self.nstep+1)
        for i in range(self.nstep+1):
            KE[i] = 0.5*self.energy[i].Integrate()
        return KE
        
    def PEnergy(self):
        PE = np.zeros(self.nstep+1)
        if self.field_solve == 'fast':
            for i in range(self.nstep+1):
                pe = self.rho[i]*self.phi[i]
                PE[i] = 0.5*self.chrg*pe.Integrate()
        else:
            for i in range(self.nstep+1):
                pe = self.rho[i].RegularGrid(self.ncxf,self.ncyf)*self.phi[i]
                PE[i] = 0.5*self.chrg*np.sum(pe)*self.dxf*self.dyf
        return PE
        
    def AnimateResult(self,name,sizex,sizey,npx,npy,tp_per_real_sec,max_framerate,flabelstart,zmax=1.,zmin=0.,filt=False):
        fig = plt.figure(figsize=(sizex,sizey))
        ax1 = fig.add_subplot(111,autoscale_on=False,xlim=(0,self.DomLenX), ylim=(0,self.DomLenY), aspect='equal')
        #ax2 = fig.add_subplot(132,autoscale_on=False,xlim=(0,self.DomLenX), ylim=(0,self.DomLenY),aspect='equal')
        #ax3 = fig.add_subplot(133,autoscale_on=False,xlim=(0,self.DomLenX), ylim=(0,self.DomLenY),aspect='equal')
        #ax4 = fig.add_subplot(224,autoscale_on=False,xlim=(0,self.DomLen), ylim=(np.amin(self.E),np.amax(self.E)))
        x = np.linspace(0.,self.DomLenX,npx,endpoint=False)
        y = np.linspace(0.,self.DomLenY,npy,endpoint=False)
        Y, X = np.meshgrid(x,y)        
        if zmax > 2.*np.amax(self.rho[0].RegularGrid(npx,npy)):
            zmax = np.amax(self.rho[0].RegularGrid(npx,npy))
        
        quad1 = ax1.pcolormesh(X,Y,self.rho[0].RegularGrid(npx,npy),vmax=zmax,vmin=zmin,cmap='jet')#; quad2 = ax2.pcolormesh(X,Y,self.energy[0].RegularGrid())
        #quad3 = ax3.pcolormesh(X,Y,self.phi[0].RegularGrid());
        
        ax1.set_title('Density',fontsize=14)#; ax2.set_title('KE density',fontsize=14)
        #ax3.set_title('Electric Potential',fontsize=14)
        
        #ax3.set_xlabel(r'$x/\lambda_D$',fontsize=13); ax4.set_xlabel(r'$x/\lambda_D$',fontsize=13)
        
        ax1.set_xlabel(r'$x/\lambda_D$',fontsize=13); ax1.set_ylabel(r'$y/\lambda_D$',fontsize=13)
        
        mult = int(1); speed = int(tp_per_real_sec*self.rho.shape[0]/self.T)
        while speed > max_framerate:
            speed = int(speed/2.)
            mult *= 2
            
        numframes = int(self.rho.shape[0]/mult)
        print('Number of frames in movie: ' + str(numframes))
        print('Framerate = ' + str(speed))
        
        
        def init():
            quad1.set_array(np.ndarray([]))#; quad2.set_array(np.ndarray([]))
            #quad3.set_data(np.ndarray([]))
            return quad1, #quad2, quad3
        
        def animate(i):
            r = self.rho[mult*i].RegularGridBicubic(npx,npy)#; e = self.energy[i].RegularGrid()
            if filt:            
                r = FIL.BinomialFilter2D(r)
            #p = self.phi[i].RegularGrid()
            quad1.set_array(r[:-1,:-1].ravel())#; quad2.set_array(e[:-1,:-1].ravel())
            #quad3.set_array(p[:-1,:-1].ravel())
            l = i + flabelstart
            fr = 'frame%03d.png' %l
            framename = 'MovieFrames/' + name + fr
            fig.savefig(framename)
            return quad1, #quad2, quad3
            
        fullname = name + '.mp4'
        
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=numframes, blit=False)
        anim.save(fullname, writer=animation.FFMpegWriter(fps=speed))
        #anim.save(fullname, writer=animation.FFMpegWriter(), fps=speed)
