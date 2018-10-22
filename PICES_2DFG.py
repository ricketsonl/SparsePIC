# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:45:53 2016

@author: ricketsonl
"""

import numpy as np
#from numpy.linalg import norm
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()},reload_support=True)
import InterpolationRoutines as IR
import SpectralOps as SO
#from scipy.optimize import newton_krylov
import matplotlib.pyplot as plt
from matplotlib import animation
#plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'


class PICES2DFG:
    chrg = -1.
    space_norm = 1; time_norm = np.inf # Sets the space- and time-norms for variances
    imptol = 1.e-10 # Tolerance for Newton-Krylov iteration
    B = 10.
    def __init__(self,T_final,domx,domy,numcellsx,numcellsy,numsteps,idata,Npi,varwts=False,RHS=0.):
        self.T = T_final
        self.ncelx = numcellsx; self.ncely = numcellsy
        self.nstep = numsteps
        self.dt = T_final/numsteps
        self.DomLenX = domx; self.DomLenY = domy
        self.dx = domx/numcellsx; self.dy = domy/numcellsy

        self.time = 0.
        
        self.IDatGen = idata

        self.x, self.v = self.IDatGen(Npi,domx,domy)
        if np.amin(self.x) < 0. or np.amax(self.x[:,0]) > self.DomLenX or np.amax(self.x[:,1]) > self.DomLenY:
            print('Warning: Some particles initialized outside domain')

        self.rho = np.zeros((numsteps+1,self.ncelx,self.ncely)); self.u = np.zeros((numsteps+1,self.ncelx,self.ncely,2))
        self.energy = np.zeros((numsteps+1,self.ncelx,self.ncely)); self.E = np.zeros((numsteps+1,self.ncelx,self.ncely,2))
        
        self.Epar = np.zeros((self.ncelx,self.ncely,2)); self.phi = np.zeros((numsteps+1,self.ncelx,self.ncely))
        
        self.kenergy = np.zeros(numsteps+1)
        
        self.space_avg = 1.; self.time_avg = 1. # Parameters to turn norms into averages over cells
        # if not taking infinity-norm
        if self.space_norm != np.inf:
            self.space_avg = self.ncelx*self.ncely
        if self.time_norm != np.inf:
            self.time_avg = self.nstep

        ## If you're allowing for varying particle weights, setup initial particle weights 
        ## and store RHS forcing function in Vlasov equation
        ## Note: If you're going to vary particle weights, you MUST specify RHS in the constructor 
        ## as a function of nparrays x and v, which have shape (Npi,2), and of t.  That is, 
        ## RHS = RHS(x,v,t)
        self.varwts = varwts
        if self.varwts:
            self.parwts = np.ones(Npi)
            self.rhs = RHS
            
    def IG(self,vee): ## Grid interpolation wrapper function
        F = IR.InterpGrid2D(self.x,vee,self.dx,self.dy,self.ncelx,self.ncely)
        return F
        
    def IGVec(self,vee):
        F = np.zeros((self.ncelx,self.ncely,2))
        F[:,:,0] = IR.InterpGrid2D(self.x,vee[:,0],self.dx,self.dy,self.ncelx,self.ncely)
        F[:,:,1] = IR.InterpGrid2D(self.x,vee[:,1],self.dx,self.dy,self.ncelx,self.ncely)
        return F        
        
    def IPVec(self,EE):
        F = np.zeros_like(self.x)
        F[:,0] = IR.InterpPar2D(self.x,EE[:,:,0],self.dx,self.dy,self.ncelx,self.ncely)
        F[:,1] = IR.InterpPar2D(self.x,EE[:,:,1],self.dx,self.dy,self.ncelx,self.ncely)
        return F
        
    def Div(self,F):
        Fx_y, Fx_x = SO.SpectralDerivative2D(F[:,:,0],self.DomLenX,self.DomLenY)
        Fy_y, Fy_x = SO.SpectralDerivative2D(F[:,:,1],self.DomLenX,self.DomLenY)
        return Fx_x+Fy_y
        
    def resid(self,Lphinew,i):
        phinew = SO.Poisson2Dperiodic(Lphinew,self.DomLenX,self.DomLenY)
        Enew = np.zeros((self.ncelx,self.ncely,2))
        Enew[:,:,1], Enew[:,:,0] = SO.SpectralDerivative2D(-1.*phinew,self.DomLenX,self.DomLenY)
        self.Epar = self.IPVec(0.5*(Enew+self.E[i]))
        vnew = self.v + self.dt*self.chrg*self.Epar
        j = self.chrg*self.IGVec(0.5*(vnew+self.v))
        return Lphinew - self.LaplacePhi - self.dt*self.DomLenX*self.DomLenY*self.Div(j)
    
    def PushPars(self,effE):
        self.Epar = self.IPVec(effE)
        self.v += 0.5*self.dt*self.chrg*self.Epar
        t = self.chrg*self.B*self.dt/2.0; s = 2.0*t/(1.0 + t*t); c = (1-t*t)/(1+t*t)
        vnewx = self.v[:,0]*c - self.v[:,1]*s
        vnewy = self.v[:,0]*s + self.v[:,1]*c
        self.v[:,0] = vnewx; self.v[:,1] = vnewy
        self.v += 0.5*self.dt*self.chrg*self.Epar
        self.x += self.dt*self.v
        self.x[:,0] %= self.DomLenX; self.x[:,1] %= self.DomLenY

        ## If varying weights, update them here
        if self.varwts:
            self.parwts += self.dt*self.rhs(self.x,self.v,self.time)
        
    def InterpGridHydros(self,i):
        if self.varwts:
            self.rho[i] = self.IG(self.parwts)
        else: 
            self.rho[i] = self.IG(np.ones(self.x.shape[0]))
        #self.u[i] = self.IGVec(self.v)
        #self.energy[i] = self.IG(self.v[:,0]**2 + self.v[:,1]**2)
        self.kenergy[i] = 0.5*np.sum(self.v[:,0]**2 + self.v[:,1]**2)/self.x.shape[0]
        
    def Initialize(self,N):
        self.x, self.v = self.IDatGen(N,self.DomLenX,self.DomLenY)
        self.InterpGridHydros(0)
        self.phi[0] = SO.Poisson2Dperiodic(self.chrg*(self.rho[0] - np.mean(self.rho[0]))*self.DomLenX*self.DomLenY,self.DomLenX,self.DomLenY)
        self.E[0,:,:,1], self.E[0,:,:,0] = SO.SpectralDerivative2D(-1.*self.phi[0],self.DomLenX,self.DomLenY)
        
    def ComputeFieldPoisson(self,i):
        self.phi[i] = SO.Poisson2Dperiodic(self.chrg*(self.rho[i] - np.mean(self.rho[i]))*self.DomLenX*self.DomLenY,self.DomLenX,self.DomLenY)
        self.E[i,:,:,1], self.E[i,:,:,0] = SO.SpectralDerivative2D(-1.*self.phi[i],self.DomLenX,self.DomLenY)
        
    def Run(self,N,notify=50):
        self.Initialize(N)
        for i in range(0,self.nstep):
            self.ComputeFieldPoisson(i)
            self.PushPars(self.E[i])
            self.InterpGridHydros(i+1)
            if i % notify == 0:
                print('Completed time-step ' + str(i) + ': Simulation time is ' + str(self.time))
            self.time += self.dt
        self.ComputeFieldPoisson(self.nstep)
        
    def KEnergy(self):
        return 0.5*np.sum(self.energy,axis=(1,2))*self.dx*self.dy
    def PEnergy(self):
        #PE = self.E[:,:,:,1]**2 + self.E[:,:,:,0]**2
        #return 0.5*np.sum(PE,axis=(1,2))*self.dx*self.dy/(self.DomLenX*self.DomLenY)
        return 0.5*np.sum(self.phi*self.rho,axis=(1,2))*self.chrg*self.dx*self.dy
        
        
    def AnimateResult(self,name,sizex,sizey,tp_per_real_sec,max_framerate,flabelstart,zmax=1., zmin=0.):
        fig = plt.figure(figsize=(sizex,sizey))
        ax1 = fig.add_subplot(111,autoscale_on=False,xlim=(0,self.DomLenX), ylim=(0,self.DomLenY), aspect='equal')
        #ax2 = fig.add_subplot(132,autoscale_on=False,xlim=(0,self.DomLenX), ylim=(0,self.DomLenY),aspect='equal')
        #ax3 = fig.add_subplot(133,autoscale_on=False,xlim=(0,self.DomLenX), ylim=(0,self.DomLenY),aspect='equal')
        #ax4 = fig.add_subplot(224,autoscale_on=False,xlim=(0,self.DomLen), ylim=(np.amin(self.E),np.amax(self.E)))
        x = np.linspace(0.,self.DomLenX,self.ncelx,endpoint=False)
        y = np.linspace(0.,self.DomLenY,self.ncely,endpoint=False)
        Y, X = np.meshgrid(x,y) 
        if zmax > 2.*np.amax(self.rho[0]):
            zmax = np.amax(self.rho[0])
        #zmax = 0.8
        
        quad1 = ax1.pcolormesh(X,Y,self.rho[0],vmax=zmax,vmin=zmin,cmap='jet')#; quad2 = ax2.pcolormesh(X,Y,self.energy[0].RegularGrid())
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
            r = self.rho[mult*i]#; e = self.energy[i].RegularGrid()
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
