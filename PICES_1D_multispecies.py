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


class PICES1D:
    chrge = -1.
    chrgi = +1.
    space_norm = 1; time_norm = np.inf # Sets the space- and time-norms for variances
    imptol = 1.e-10 # Tolerance for Newton-Krylov iteration
    def __init__(self,T_final,domlen,numcells,numsteps,idata,npi,varwts=False):
        self.T = T_final
        self.ncel = numcells
        self.nstep = numsteps
        self.dt = T_final/numsteps
        self.DomLen = domlen
        self.dx = domlen/numcells
        self.npi = npi
        self.EfieldInfErr = -1.0
        self.EfieldL2Err = -1.0

        self.x = np.linspace(0.,self.DomLen,num=self.ncel,endpoint=False)

        self.ex1out = open("ex1out.txt","a+")
        self.ex1out.write("ex1 = [")
        self.cp1out = open("cp1out.txt","a+")
        self.cp1out.write("cp1 = [")

        self.time = 0.
        
        self.IDatGen = idata

        self.xi, self.vi, self.xe, self.ve = self.IDatGen(self.npi,self.DomLen)
        self.partwtse = np.ones(self.npi); self.parwtsi = np.ones(self.npi)
        # IMPOSE charge tracking for individual particles.
        if np.amin(self.xe) < 0. or np.amax(self.xe) > self.DomLen:
            print('Warning: Some particles initialized outside domain')

        if np.amin(self.xi) < 0. or np.amax(self.xi) > self.DomLen:
            print('Warning: Some particles initialized outside domain')


        #initialize grid variables for each time. (PT)
        self.rho = np.zeros((numsteps+1,self.ncel)); self.u = np.zeros((numsteps+1,self.ncel))
        self.energy = np.zeros((numsteps+1,self.ncel)); self.E = np.zeros((numsteps+1,self.ncel))
        
        self.Epar = np.zeros(self.ncel); self.phi = np.zeros((numsteps+1,self.ncel))
        
        self.kenergy = np.zeros(numsteps+1)
        
        self.space_avg = 1.; self.time_avg = 1. # Parameters to turn norms into averages over cells
        # if not taking infinity-norm
        if self.space_norm != np.inf:
            self.space_avg = self.ncel
        if self.time_norm != np.inf:
            self.time_avg = self.nstep

        ## If you're allowing for varying particle weights, setup initial particle weights 
        ## and store RHS forcing function in Vlasov equation
        ## Note: If you're going to vary particle weights, you MUST specify RHS in the constructor 
        ## as a function of nparrays x and v, which have shape (Npi,2), and of t.  That is, 
        ## RHS = RHS(x,v,t)
        self.varwts = varwts
        if self.varwts:
#            self.parwtse = np.ones(Npi)
#            self.parwtsi = np.ones(Npi)
            self.rhse = RHSe
            self.rhsi = RHSi
            self.efield = Efield

    def IG(self,wte,wti): ## Grid interpolation wrapper function
        F = self.chrgi*IR.InterpGrid(self.xi,wti,self.dx,self.ncel)
        F = self.chrge*IR.InterpGrid(self.xe,wte,self.dx,self.ncel)
        F = F + self.chrgi*IR.InterpGrid(self.xi,wti,self.dx,self.ncel)
        return F
        
    def IGVec(self,vee):
        F = np.zeros((self.ncelx,self.ncely,2))
        F[:,:,0] = IR.InterpGrid2D(self.x,vee[:,0],self.dx,self.dy,self.ncelx,self.ncely)
        F[:,:,1] = IR.InterpGrid2D(self.x,vee[:,1],self.dx,self.dy,self.ncelx,self.ncely)
        return F        
        
#    def IPVec(self,EE):
#        F = np.zeros_like(self.x)
#        F[:,0] = IR.InterpPar2D(self.x,EE[:,:,0],self.dx,self.dy,self.ncelx,self.ncely)
#        F[:,1] = IR.InterpPar2D(self.x,EE[:,:,1],self.dx,self.dy,self.ncelx,self.ncely)
#        return F

    def IPVec(self,EE):
        Fe = np.zeros_like(self.xe)
        Fe = IR.InterpPar(self.xe,EE,self.dx,self.ncel)

        Fi = np.zeros_like(self.xi)
        Fi = IR.InterpPar(self.xi,EE,self.dx,self.ncel)

        return Fe, Fi
        
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
        # Doubled everything up, to account for splitting of particles into electron and ions.
        self.Epare, self.Epari = self.IPVec(effE)

        self.ve += self.dt*self.chrge*self.Epare
        self.vi += self.dt*self.chrgi*self.Epari


        ## If varying weights, update them here
        if self.varwts:
            self.xe += 0.5*self.dt*self.ve
            self.xi += 0.5*self.dt*self.vi

            self.xe[:,0] %= self.DomLenX; self.xe[:,1] %= self.DomLenY
            self.xi[:,0] %= self.DomLenX; self.xi[:,1] %= self.DomLenY
            
            self.parwtsi += self.dt*self.rhsi(self.xi,self.vi,self.time+0.5*self.dt,self.chrge,self.DomLenX,self.DomLenY)/self.f0i
            self.parwtse += self.dt*self.rhse(self.xe,self.ve,self.time+0.5*self.dt,self.chrge,self.DomLenX,self.DomLenY)/self.f0e
            
            self.xe += 0.5*self.dt*self.ve
            self.xi += 0.5*self.dt*self.vi

            self.xe[:,0] %= self.DomLenX; self.xe[:,1] %= self.DomLenY
            self.xi[:,0] %= self.DomLenX; self.xi[:,1] %= self.DomLenY
        else:
            self.xe += self.dt*self.ve
            self.xi += self.dt*self.vi

            self.xe %= self.DomLen
            self.xi %= self.DomLen
            

    def InterpGridHydros(self,i):
        if self.varwts:
            self.rho[i] = self.IG(self.parwtse, self.parwtsi)
        else: 
            self.rho[i] = self.IG(np.ones(self.xe.shape[0]), np.ones(self.xi.shape[0]))
        #self.u[i] = self.IGVec(self.v)
        #self.energy[i] = self.IG(self.v[:,0]**2 + self.v[:,1]**2)
        self.kenergy[i] = 0.5*np.sum(self.ve**2)/self.xe.shape[0]
        self.kenergy[i] += 0.5*np.sum(self.vi**2)/self.xi.shape[0]

        
    def Initialize(self,N):
        self.xi, self.vi, self.xe, self.ve = self.IDatGen(N,self.DomLen)
        self.parwtse = np.ones_like(self.xe); self.parwtsi = np.ones_like(self.xi)
        self.InterpGridHydros(0)
#        self.phi[0] = SO.Poisson2Dperiodic(self.chrge*(self.rho[0] - np.mean(self.rho[0]))*self.DomLenX*self.DomLenY,self.DomLenX,self.DomLenY)
        self.phi[0] = SO.Poisson1Dperiodic(self.rho[0],self.DomLen)
        self.E[0,:] = SO.SpectralDerivative(-1.*self.phi[0],self.DomLen)
        
        ## Do an initial backward half-step to offset positions and velocities in time
        self.Epare, self.Epari = self.IPVec(self.E[0])
        dt_tmp = -1.*0.5*self.dt

        self.vi += dt_tmp*self.chrgi*self.Epari
        self.ve += dt_tmp*self.chrge*self.Epare

    def ComputeFieldPoisson(self,i):
#        self.phi[i] = SO.Poisson2Dperiodic(self.chrge*(self.rho[i] - np.mean(self.rho[i]))*self.DomLenX*self.DomLenY,self.DomLenX,self.DomLenY)
        self.phi[i] = SO.Poisson1Dperiodic(self.rho[i],self.DomLen)
        self.E[i,:] = SO.SpectralDerivative(-1.*self.phi[i],self.DomLen)

    def TestEField(self,timeIndex):
        Etest = np.zeros((self.ncelx,self.ncely,2))
        rhoTest = np.zeros((self.ncelx,self.ncely))
        l2err = np.zeros(2)
        Time = self.time
        print('Time = ' + str(Time))
        x = 0.
        for i in range(0,self.ncelx):
            y = 0.
            for j in range(0,self.ncely):
                Etest[i,j,:] = self.efield((x,y),Time,self.DomLenX,self.DomLenY)
                rhoTest[i,j] = 4.*np.pi**2*np.sin(np.pi*Time)*(np.sin(2.*np.pi*x/self.DomLenX)/self.DomLenX**2 + np.cos(2.*np.pi*y/self.DomLenY)/self.DomLenY**2)
                y += self.dy
                l2err[0] += np.abs(Etest[i,j,0] - self.E[timeIndex,i,j,0])**2
                l2err[1] += np.abs(Etest[i,j,1] - self.E[timeIndex,i,j,1])**2
            x += self.dx
        print('x,y = (' + str(x) + ',' + str(y) + ')')
        inferr = np.max(np.abs(Etest[:,:,0] - self.E[timeIndex,:,:,0]))
        inferr = np.maximum(inferr,np.max(np.abs(Etest[:,:,1] - self.E[timeIndex,:,:,1])))
        self.EfieldInfErr = np.maximum(inferr,self.EfieldInfErr)
        err = np.sqrt(np.maximum(l2err[0],l2err[1]))
        self.EfieldL2Err = np.maximum(err,self.EfieldInfErr)
        self.ex1out.write(str(Etest[0,0,0]) + "\n")
        self.cp1out.write(str(self.E[timeIndex,0,0,0]) + "\n")
        print('computed E')
        print(self.E[timeIndex,:,:,0])
        print('manufactured E')
        print(Etest[:,:,0])

        
        
    def Run(self,N,notify=50):
        self.Initialize(N) #Set up initial particle x and v.  Compute initial electric field. #####
#        self.TestEField(0)
        for i in range(0,self.nstep):
#        for i in range(0,2):
            self.ComputeFieldPoisson(i) #Compute new electric field. #### NO CHANGE NEEDED
#            self.TestEField(i)
            self.PushPars(self.E[i]) #Update particle locations.  Includes E-field interpolation. #####
            self.InterpGridHydros(i+1) #Compute Rho. ######
            if i % notify == 0:
                print('Completed time-step ' + str(i) + ': Simulation time is ' + str(self.time))
                #print('parwtse[0] = ' + str(self.parwtse[0]))
            #self.TestEField(i)
            self.time += self.dt
#            self.TestEField(i)
        self.ComputeFieldPoisson(self.nstep)
       # Fe, Fi = self.computeEDF((0.5,0.5),(-0.02,0.03))
       # Fe = self.computeCDF((0.5,0.5),(-0.02,0.03),self.time,self.manSolne)
       # Fi = self.computeCDF((0.5,0.5),(-0.02,0.03),self.time,self.manSolni)
#        self.computeErrors(25)
        self.ex1out.write("];")
        self.cp1out.write("];")
        self.ex1out.close()
        self.cp1out.close()
        print('Infinity Norm Error in E = ' + str(self.EfieldInfErr))
        print('L2spaceInftime Error in E = ' + str(self.EfieldL2Err))
        
    def KEnergy(self):
        return 0.5*np.sum(self.energy,axis=(1))*self.dx
    def PEnergy(self):
        #PE = self.E[:,:,:,1]**2 + self.E[:,:,:,0]**2
        #return 0.5*np.sum(PE,axis=(1,2))*self.dx*self.dy/(self.DomLenX*self.DomLenY)
        return 0.5*np.sum(self.phi*self.rho,axis=(1))*self.dx
        
        
    def AnimateResult(self,name,sizex,sizey,tp_per_real_sec,max_framerate,flabelstart,zmax=1., zmin=0.):
        fig = plt.figure(figsize=(sizex,sizey))
        ax1 = fig.add_subplot(111,autoscale_on=False,xlim=(0,self.DomLen), ylim=(1.2*np.amin(self.rho[0]),1.2*np.amax(self.rho[0])))
        #ax2 = fig.add_subplot(132,autoscale_on=False,xlim=(0,self.DomLenX), ylim=(0,self.DomLenY),aspect='equal')
        #ax3 = fig.add_subplot(133,autoscale_on=False,xlim=(0,self.DomLenX), ylim=(0,self.DomLenY),aspect='equal')
        #ax4 = fig.add_subplot(224,autoscale_on=False,xlim=(0,self.DomLen), ylim=(np.amin(self.E),np.amax(self.E)))
        x = np.linspace(0.,self.DomLen,self.ncel,endpoint=False)
        if zmax > 2.*np.amax(self.rho[0]):
            zmax = np.amax(self.rho[0])
        #zmax = 0.8
        
        #quad1 = ax1.pcolormesh(X,Y,self.rho[0],vmax=zmax,vmin=zmin,cmap='jet')#; quad2 = ax2.pcolormesh(X,Y,self.energy[0].RegularGrid())
        quad1, = ax1.plot(x,self.rho[0])
        
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
            quad1.set_data(x,np.ndarray([]))#; quad2.set_array(np.ndarray([]))
            #quad3.set_data(np.ndarray([]))
            return quad1, #quad2, quad3
        
        def animate(i):
            r = self.rho[mult*i]#; e = self.energy[i].RegularGrid()
            #p = self.phi[i].RegularGrid()
            quad1.set_data(x,r)#; quad2.set_array(e[:-1,:-1].ravel())
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
