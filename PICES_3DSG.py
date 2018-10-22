# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:27:22 2016

@author: ricketsonl
"""
# Class for doing 3-D electrostatic, explicit PIC on sparse grids

import numpy as np
import SparseGridEff as SG
import Rotation as ROT
import matplotlib.pyplot as plt
#import InterpolationRoutines_MV as IRMV
#import SpectralOps as SO
from matplotlib import animation
#plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'


class PICES3DSG:
    chrg = -1.
    space_norm = 1; time_norm = np.inf # Sets the space- and time-norms for variances
    def __init__(self,T_final,domx,domy,domz,numcellsx,numsteps,idata,Bfield_in,Npi,ifEext=False,Eext=0.,rho_overall=1.,nonunif_ions=False,ion_dens=0.):
        self.T = T_final
        self.n = int(np.log2(numcellsx-0.01))+1
        self.ncelx = 2**self.n; self.ncely = 2**self.n; self.ncelz = 2**self.n
        self.nstep = numsteps
        self.dt = T_final/numsteps
        self.DomLenX = domx; self.DomLenY = domy; self.DomLenZ = domz
        self.dx = domx/self.ncelx; self.dy = domy/self.ncely; self.dz = domz/self.ncelz

        self.time = 0.
        
        self.IDatGen = idata
        self.Bfield = Bfield_in
        self.E_imposed = Eext
        self.rho_mult = rho_overall
        self.ifEext = ifEext
        
        self.ifNonUnifIons = nonunif_ions
        if nonunif_ions:
            self.ion_dens = SG.SparseGridEff3D(domx,domy,domz,self.n)
        
            xi = ion_dens(Npi) 
        
            self.ion_dens.InterpolateData(xi,np.ones(Npi))
        else:
            self.ion_dens=0.
        

        self.x, self.v = self.IDatGen(Npi,domx,domy,domz)
        if np.amin(self.x) < 0. or np.amax(self.x[:,0]) > self.DomLenX or np.amax(self.x[:,1]) > self.DomLenY or np.amax(self.x[:,2]) > self.DomLenZ:
            print('Warning: Some particles initialized outside domain')
            
        self.parwts = np.ones(Npi)

        vSparseGrid = np.vectorize(SG.SparseGridEff3D)
        self.rho = vSparseGrid(domx*np.ones(numsteps+1),domy*np.ones(numsteps+1),domz*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int))    
        #self.ux = vSparseGrid(domx*np.ones(numsteps+1),domy*np.ones(numsteps+1),domz*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int))
        #self.uy = vSparseGrid(domx*np.ones(numsteps+1),domy*np.ones(numsteps+1),domz*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int)) 
        #self.energy = vSparseGrid(domx*np.ones(numsteps+1),domy*np.ones(numsteps+1),domz*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int)) 
        self.Ex = vSparseGrid(domx*np.ones(numsteps+1),domy*np.ones(numsteps+1),domz*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int)) 
        self.Ey = vSparseGrid(domx*np.ones(numsteps+1),domy*np.ones(numsteps+1),domz*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int))
        self.Ez = vSparseGrid(domx*np.ones(numsteps+1),domy*np.ones(numsteps+1),domz*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int))
        self.phi = vSparseGrid(domx*np.ones(numsteps+1),domy*np.ones(numsteps+1),domz*np.ones(numsteps+1),self.n*np.ones(numsteps+1,dtype=int)) 
        
        self.KE = np.zeros(numsteps+1)
        
        self.Epar = np.zeros((Npi,3))
    
    def PushPars(self,i):
        self.Epar[:,0] = self.Ex[i].EvaluateAt(self.x)
        self.Epar[:,1] = self.Ey[i].EvaluateAt(self.x)
        self.Epar[:,2] = self.Ez[i].EvaluateAt(self.x)
        if self.ifEext:
            Eext = self.E_imposed(self.x)
        else:
            Eext = 0.
        
        self.v += 0.5*self.dt*self.chrg*(self.Epar + Eext)
        #vnewx = self.v[:,0]*np.cos(self.chrg*self.B*self.dt) + self.v[:,1]*np.sin(self.chrg*self.B*self.dt)
        #vnewy = -self.v[:,0]*np.sin(self.chrg*self.B*self.dt) + self.v[:,1]*np.cos(self.chrg*self.B*self.dt)
        #self.v[:,0] = vnewx; self.v[:,1] = vnewy
        Bpar = self.Bfield(self.x)
        self.v = ROT.Brotate(self.v,Bpar,self.chrg,self.dt)
        
        self.v += 0.5*self.dt*self.chrg*(self.Epar + Eext)
        
        self.x += self.dt*self.v
        self.x[:,0] %= self.DomLenX; self.x[:,1] %= self.DomLenY; self.x[:,2] %= self.DomLenZ
        
    def InterpGridHydros(self,i):
        self.rho[i].InterpolateData(self.x,self.parwts)
        #self.ux[i].InterpolateData(self.x,self.v[:,0])
        #self.uy[i].InterpolateData(self.x,self.v[:,1])
        #self.energy[i].InterpolateData(self.x,self.v[:,0]**2 + self.v[:,1]**2)
        self.KE[i] = 0.5*np.sum(self.v[:,0]*self.v[:,0] + self.v[:,1]*self.v[:,1] + self.v[:,2]*self.v[:,2])/self.x.shape[0]
        
    def Initialize(self):
        self.InterpGridHydros(0)
        #rho_reg = self.rho[0].RegularGrid(self.ncelx,self.ncely,self.ncelz)
        PoissRHS = self.chrg*(self.rho[0]-self.rho[0].Mean())*self.DomLenX*self.DomLenY*self.DomLenZ
        self.phi[0] = PoissRHS.SparsePoisson()
        self.Ex[0], self.Ey[0], self.Ez[0] = self.phi[0].SparseDerivative()
        self.Ex[0] = -1.*self.Ex[0]; self.Ey[0] = -1.*self.Ey[0]; self.Ez[0] = -1.*self.Ez[0]
        
    def ComputeFieldPoisson(self,i):
        #rho_reg = self.rho[i].RegularGrid(self.ncelx,self.ncely,self.ncelz)
        if self.ifNonUnifIons:
            PoissRHS = self.rho_mult*self.chrg*(self.rho[i] - self.ion_dens)*self.DomLenX*self.DomLenY*self.DomLenZ
        else:
            PoissRHS = self.rho_mult*self.chrg*(self.rho[i] - self.rho[i].Mean())*self.DomLenX*self.DomLenY*self.DomLenZ
        self.phi[i] = PoissRHS.SparsePoisson()
        self.Ex[i], self.Ey[i], self.Ez[i] = self.phi[i].SparseDerivative()
        self.Ex[i] = -1.*self.Ex[i]; self.Ey[i] = -1.*self.Ey[i]; self.Ez[i] = -1.*self.Ez[i]
        
    def Run(self,notify=50):
        self.Initialize()
        for i in range(0,self.nstep):
            self.ComputeFieldPoisson(i)
            self.PushPars(i)
            self.InterpGridHydros(i+1)
            if i % notify == 0:
                print('Completed time-step ' + str(i) + ': Simulation time is ' + str(self.time))
            self.time += self.dt
        self.ComputeFieldPoisson(self.nstep)
        
    def KEnergy(self):
        KE = np.zeros(self.nstep+1)
        for i in range(self.nstep+1):
            KE[i] = 0.5*self.energy[i].Integrate()
        return KE
        
    def PEnergy(self):
        PE = np.zeros(self.nstep+1)
        for i in range(self.nstep+1):
            #rho_reg = self.rho[i].RegularGrid(self.ncelx,self.ncely,self.ncelz)
            pe = self.rho[i]*self.phi[i]
            PE[i] = 0.5*self.chrg*pe.Integrate()
        return PE
        
    def AnimateResult(self,name,sizex,sizey,npx,npy,realtime_per_tp,max_framerate,zmax=1.,zmin=0.):
        fig = plt.figure(figsize=(sizex,sizey))
        ax1 = fig.add_subplot(111,autoscale_on=False,xlim=(0,self.DomLenX), ylim=(0,self.DomLenY), aspect='equal')
        #ax2 = fig.add_subplot(132,autoscale_on=False,xlim=(0,self.DomLenX), ylim=(0,self.DomLenY),aspect='equal')
        #ax3 = fig.add_subplot(133,autoscale_on=False,xlim=(0,self.DomLenX), ylim=(0,self.DomLenY),aspect='equal')
        #ax4 = fig.add_subplot(224,autoscale_on=False,xlim=(0,self.DomLen), ylim=(np.amin(self.E),np.amax(self.E)))
        x = np.linspace(0.,self.DomLenX,npx,endpoint=False)
        y = np.linspace(0.,self.DomLenY,npy,endpoint=False)
        Y, X = np.meshgrid(x,y)      
        
        plt0 = self.rho[0].Regular2DSlice('z',self.DomLenZ/2.,npx,npy)
        # Fix colormap scaling if necessary
        zmaxnew = np.amax(plt0); zminnew = np.amin(plt0)
        if zmaxnew < zmax/1.5:
            zmax = zmaxnew
        if zmaxnew - zminnew < 0.5*(zmax-zmin):
            zmin = zminnew
        
        
        quad1 = ax1.pcolormesh(X,Y,plt0,vmax=zmax,vmin=zmin)#; quad2 = ax2.pcolormesh(X,Y,self.energy[0].RegularGrid())
        #quad3 = ax3.pcolormesh(X,Y,self.phi[0].RegularGrid());
        
        ax1.set_title('Density',fontsize=14)#; ax2.set_title('KE density',fontsize=14)
        #ax3.set_title('Electric Potential',fontsize=14)
        
        #ax3.set_xlabel(r'$x/\lambda_D$',fontsize=13); ax4.set_xlabel(r'$x/\lambda_D$',fontsize=13)
        
        mult = int(1); speed = int(realtime_per_tp*self.rho.shape[0]/self.T)
        while speed > max_framerate:
            speed = int(speed/2.)
            mult *= 2
            
        numframes = int(self.rho.shape[0]/mult)
        
        
        def init():
            quad1.set_array(np.ndarray([]))#; quad2.set_array(np.ndarray([]))
            #quad3.set_data(np.ndarray([]))
            return quad1, #quad2, quad3
        
        def animate(i):
            r = self.rho[mult*i].Regular2DSlice('z',self.DomLenZ/2.,npx,npy)#; e = self.energy[i].RegularGrid()
            #p = self.phi[i].RegularGrid()
            quad1.set_array(r[:-1,:-1].ravel())#; quad2.set_array(e[:-1,:-1].ravel())
            #quad3.set_array(p[:-1,:-1].ravel())
            
            fr = 'frame%03d.png' %i
            framename = 'MovieFrames/' + name + fr
            fig.savefig(framename)            
            
            return quad1, #quad2, quad3
            
        fullname = name + '.mp4'
        
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=numframes, blit=False)
        anim.save(fullname, writer=animation.FFMpegWriter(fps=speed))
        #anim.save(fullname, writer=animation.FFMpegWriter(), fps=speed)
