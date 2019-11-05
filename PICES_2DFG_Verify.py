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


class PICES2DFGV:
    chrge = -1.
    chrgi = +1.
    space_norm = 1; time_norm = np.inf # Sets the space- and time-norms for variances
    imptol = 1.e-10 # Tolerance for Newton-Krylov iteration
    B = 10.
    def __init__(self,T_final,domx,domy,numcellsx,numcellsy,numsteps,idata,Npi,varwts=False,RHSe=0.,RHSi=0.,Efield = 0.,mSe=0.,mSi=0.):
        self.T = T_final
        self.ncelx = numcellsx; self.ncely = numcellsy
        self.nstep = numsteps
        self.dt = T_final/numsteps
        self.DomLenX = domx; self.DomLenY = domy
        self.dx = domx/numcellsx; self.dy = domy/numcellsy
        self.npi = Npi
        self.manSolne = mSe
        self.manSolni = mSi
        self.EfieldInfErr = -1.0
        self.EfieldL2Err = -1.0

        self.ex1out = open("ex1out.txt","a+")
        self.ex1out.write("ex1 = [")
        self.cp1out = open("cp1out.txt","a+")
        self.cp1out.write("cp1 = [")

        self.time = 0.
        
        self.IDatGen = idata

        self.xi, self.vi, self.parwtsi, self.f0i, self.xe, self.ve, self.parwtse, self.f0e = self.IDatGen(Npi,domx,domy, 5.)
        # IMPOSE charge tracking for individual particles.
        if np.amin(self.xe) < 0. or np.amax(self.xe[:,0]) > self.DomLenX or np.amax(self.xe[:,1]) > self.DomLenY:
            print('Warning: Some particles initialized outside domain')

        if np.amin(self.xi) < 0. or np.amax(self.xi[:,0]) > self.DomLenX or np.amax(self.xi[:,1]) > self.DomLenY:
            print('Warning: Some particles initialized outside domain')


        #initialize grid variables for each time. (PT)
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
#            self.parwtse = np.ones(Npi)
#            self.parwtsi = np.ones(Npi)
            self.rhse = RHSe
            self.rhsi = RHSi
            self.efield = Efield

    def IG(self,wte,wti): ## Grid interpolation wrapper function
        print('T=' + str(self.time))
        F = self.chrgi*IR.InterpGrid2D(self.xi,wti,self.dx,self.dy,self.ncelx,self.ncely)
#        print('Fi')
#        print(F)
        F = self.chrge*IR.InterpGrid2D(self.xe,wte,self.dx,self.dy,self.ncelx,self.ncely)
#        print('Fe')
#        print(F)
        F = F + self.chrgi*IR.InterpGrid2D(self.xi,wti,self.dx,self.dy,self.ncelx,self.ncely)
#        print('F')
#        print(F)
#        print('Sum(F)')
#        print(np.sum(F))
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
        Fe[:,0] = IR.InterpPar2D(self.xe,EE[:,:,0],self.dx,self.dy,self.ncelx,self.ncely)
        Fe[:,1] = IR.InterpPar2D(self.xe,EE[:,:,1],self.dx,self.dy,self.ncelx,self.ncely)

        Fi = np.zeros_like(self.xi)
        Fi[:,0] = IR.InterpPar2D(self.xi,EE[:,:,0],self.dx,self.dy,self.ncelx,self.ncely)
        Fi[:,1] = IR.InterpPar2D(self.xi,EE[:,:,1],self.dx,self.dy,self.ncelx,self.ncely)

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

        self.ve += 0.5*self.dt*self.chrge*self.Epare
        self.vi += 0.5*self.dt*self.chrgi*self.Epari

        te = self.chrge*self.B*self.dt/2.0; se = 2.0*te/(1.0 + te*te); ce = (1-te*te)/(1+te*te)
        ti = self.chrgi*self.B*self.dt/2.0; si = 2.0*ti/(1.0 + ti*ti); ci = (1-ti*ti)/(1+ti*ti)

        vnewxe = self.ve[:,0]*ce - self.ve[:,1]*se
        vnewxi = self.vi[:,0]*ci - self.vi[:,1]*si

        vnewye = self.ve[:,0]*se + self.ve[:,1]*ce
        vnewyi = self.vi[:,0]*si + self.vi[:,1]*ci

        self.ve[:,0] = vnewxe; self.ve[:,1] = vnewye
        self.vi[:,0] = vnewxi; self.vi[:,1] = vnewyi

        self.ve += 0.5*self.dt*self.chrge*self.Epare
        self.vi += 0.5*self.dt*self.chrgi*self.Epari



        ## If varying weights, update them here
        if self.varwts:
            #for i in range(0,self.npi): 
            #    self.parwtse[i] += self.dt*self.rhse(self.xe[i,:],self.ve[i,:],self.time,self.chrge,self.DomLenX,self.DomLenY)/self.f0e[i]
            #    self.parwtsi[i] += self.dt*self.rhsi(self.xi[i,:],self.vi[i,:],self.time,self.chrgi,self.DomLenX,self.DomLenY)/self.f0i[i]
            self.xe += 0.5*self.dt*self.ve
            self.xi += 0.5*self.dt*self.vi

            self.xe[:,0] %= self.DomLenX; self.xe[:,1] %= self.DomLenY
            self.xi[:,0] %= self.DomLenX; self.xi[:,1] %= self.DomLenY
            
            self.parwtsi += self.dt*self.rhsi(self.xi,self.vi,self.time,self.chrge,self.DomLenX,self.DomLenY)/self.f0i
            self.parwtse += self.dt*self.rhse(self.xe,self.ve,self.time,self.chrge,self.DomLenX,self.DomLenY)/self.f0e
            
            self.xe += 0.5*self.dt*self.ve
            self.xi += 0.5*self.dt*self.vi

            self.xe[:,0] %= self.DomLenX; self.xe[:,1] %= self.DomLenY
            self.xi[:,0] %= self.DomLenX; self.xi[:,1] %= self.DomLenY
        else:
            self.xe += self.dt*self.ve
            self.xi += self.dt*self.vi

            self.xe[:,0] %= self.DomLenX; self.xe[:,1] %= self.DomLenY
            self.xi[:,0] %= self.DomLenX; self.xi[:,1] %= self.DomLenY
            

    def computeEDF(self, x, v):
        Fe = np.zeros(16); Fi = np.zeros(16)
        wisum = np.sum(self.parwtse)
        wesum = np.sum(self.parwtsi)
        for i in range(0,self.npi):
           hxe = (x[0] - self.xe[i,0]) >= 0
           hye = (x[1] - self.xe[i,1]) >= 0
           hxi = (x[0] - self.xi[i,0]) >= 0
           hyi = (x[1] - self.xi[i,1]) >= 0
           
           hue = (v[0] - self.ve[i,0]) >= 0
           hve = (v[1] - self.ve[i,1]) >= 0
           hui = (v[0] - self.vi[i,0]) >= 0
           hvi = (v[1] - self.vi[i,1]) >= 0
           
           hxeb = 1 - hxe
           hyeb = 1 - hye
           hxib = 1 - hxi
           hyib = 1 - hyi
           
           hueb = 1 - hue
           hveb = 1 - hve
           huib = 1 - hui
           hvib = 1 - hvi
           
           whate = self.parwtse[i]/wesum
           whati = self.parwtsi[i]/wisum
           
           Fe[0] = Fe[0]   + whate*(hxe*hye*hue*hve);
           Fe[1] = Fe[1]   + whate*(hxe*hye*hue*hveb);
           Fe[2] = Fe[2]   + whate*(hxe*hye*hueb*hveb);
           Fe[3] = Fe[3]   + whate*(hxe*hyeb*hue*hveb);
           Fe[4] = Fe[4]   + whate*(hxe*hyeb*hueb*hveb);
           Fe[5] = Fe[5]   + whate*(hxe*hye*hueb*hve);
           Fe[6] = Fe[6]   + whate*(hxe*hyeb*hueb*hve);
           Fe[7] = Fe[7]   + whate*(hxe*hyeb*hue*hve);
           Fe[8] = Fe[8]   + whate*(hxeb*hye*hue*hve);
           Fe[9] = Fe[9]   + whate*(hxeb*hye*hue*hveb);
           Fe[10] = Fe[10] + whate*(hxeb*hye*hueb*hveb);
           Fe[11] = Fe[11] + whate*(hxeb*hyeb*hue*hveb);
           Fe[12] = Fe[12] + whate*(hxeb*hyeb*hueb*hveb);
           Fe[13] = Fe[13] + whate*(hxeb*hye*hueb*hve);
           Fe[14] = Fe[14] + whate*(hxeb*hyeb*hueb*hve);
           Fe[15] = Fe[15] + whate*(hxeb*hyeb*hue*hve);
           
           Fi[0] = Fi[0]   + whati*(hxi*hyi*hui*hvi);
           Fi[1] = Fi[1]   + whati*(hxi*hyi*hui*hvib);
           Fi[2] = Fi[2]   + whati*(hxi*hyi*huib*hvib);
           Fi[3] = Fi[3]   + whati*(hxi*hyib*hui*hvib);
           Fi[4] = Fi[4]   + whati*(hxi*hyib*huib*hvib);
           Fi[5] = Fi[5]   + whati*(hxi*hyi*huib*hvi);
           Fi[6] = Fi[6]   + whati*(hxi*hyib*huib*hvi);
           Fi[7] = Fi[7]   + whati*(hxi*hyib*hui*hvi);
           Fi[8] = Fi[8]   + whati*(hxib*hyi*hui*hvi);
           Fi[9] = Fi[9]   + whati*(hxib*hyi*hui*hvib);
           Fi[10] = Fi[10] + whati*(hxib*hyi*huib*hvib);
           Fi[11] = Fi[11] + whati*(hxib*hyib*hui*hvib);
           Fi[12] = Fi[12] + whati*(hxib*hyib*huib*hvib);
           Fi[13] = Fi[13] + whati*(hxib*hyi*huib*hvi);
           Fi[14] = Fi[14] + whati*(hxib*hyib*huib*hvi);
           Fi[15] = Fi[15] + whati*(hxib*hyib*hui*hvi);
        return Fe, Fi
           
    def monteCarloInt(self, fun, x, v, t, a, b, Ntest):
        xtest = np.zeros(2); vtest = np.zeros(2)
        sum = 0
        V = (b[0]-a[0])*(b[1]-a[1])*(b[2]-a[2])*(b[3]-a[3])
        for i in range(0, Ntest):
            xtest[0] = np.random.uniform(a[0],b[0],1)
            xtest[1] = np.random.uniform(a[1],b[1],1)
            vtest[0] = np.random.uniform(a[2],b[2],1)
            vtest[1] = np.random.uniform(a[3],b[3],1)
            q = fun(x,v,t,self.DomLenX,self.DomLenY)
            sum = sum + fun(xtest,vtest,t,self.DomLenX, self.DomLenY)
        sum = sum*V/Ntest    
        return sum        
            
    def computeCDF(self, x, v, t, fun):
        xlim = np.zeros(2); ylim = np.zeros(2); ulim = np.zeros(2); vlim = np.zeros(2)
        F = np.zeros(16)

        Ntest = 30000
        xlim[0] = 0.; xlim[1] = self.DomLenX
        ylim[0] = 0.; ylim[1] = self.DomLenY
        ulim[0] = -15.; ulim[1] = 15.
        vlim[0] = -15.; vlim[1] = 15.
        

        #1(0,0,0,0) 
        a = (xlim[0], ylim[0], ulim[0], vlim[0])
        b = (x[0],x[1],v[0],v[1])
        F[0] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #2(0,0,0,1)
        a = (xlim[0], ylim[0], ulim[0], v[1])
        b = (x[0],x[1],v[0],vlim[1])
        F[1] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #3(0,0,1,1)
        a = (xlim[0], ylim[0], v[0], v[1])
        b = (x[0],x[1],ulim[1],vlim[1])
        F[2] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #4(0,1,0,1)
        a = (xlim[0], x[1], ulim[0], v[1])
        b = (x[0],ylim[1],v[0],vlim[1])
        F[3] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #5(0,1,1,1)
        a = (xlim[0], x[1], v[0], v[1])
        b = (x[0],ylim[1],ulim[1],vlim[1])
        F[4] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #6(0,0,1,0)
        a = (xlim[0], ylim[0], v[0], vlim[0])
        b = (x[0],x[1],ulim[1],v[1])
        F[5] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #7(0,1,1,0)
        a = (xlim[0], x[1], v[0], vlim[0])
        b = (x[0],ylim[1],ulim[1],v[1])
        F[6] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #8(0,1,0,0) 
        a = (xlim[0], x[1], ulim[0], vlim[0])
        b = (x[0],ylim[1],v[0],v[1])
        F[7] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #9(1,0,0,0) 
        a = (x[0], ylim[0], ulim[0], vlim[0])
        b = (xlim[1],x[1],v[0],v[1])
        F[8] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #10(1,0,0,1)
        a = (x[0], ylim[0], ulim[0], v[1])
        b = (xlim[1],x[1],v[0],vlim[1])
        F[9] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #11(1,0,1,1)
        a = (x[0], ylim[0], v[0], v[1])
        b = (xlim[1],x[1],ulim[1],vlim[1])
        F[10] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #12(1,1,0,1)
        a = (x[0], x[1], ulim[0], v[1])
        b = (xlim[1],ylim[1],v[0],vlim[1])
        F[11] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #13(1,1,1,1)
        a = (x[0], x[1], v[0], v[1])
        b = (xlim[1],ylim[1],ulim[1],vlim[1])
        F[12] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #14(1,0,1,0)
        a = (x[0], ylim[0], v[0], vlim[0])
        b = (xlim[1],x[1],ulim[1],v[1])
        F[13] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #15(1,1,1,0)
        a = (x[0], x[1], v[0], vlim[0])
        b = (xlim[1],ylim[1],ulim[1],v[1])
        F[14] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)
        #16(1,1,0,0) 
        a = (x[0], x[1], ulim[0], vlim[0])
        b = (xlim[1],ylim[1],v[0],v[1])
        F[15] = self.monteCarloInt(fun, x, v, t, a, b, Ntest)

        return F 
    
    def computeErrors(self, Ntest):
        a = np.zeros(4); b = np.zeros(4)
        a[0] = 0.; b[0] = self.DomLenX
        a[1] = 0.; b[1] = self.DomLenY
        a[2] = -15.; b[2] = 15.
        a[3] = -15.; b[3] = 15.

        erre = np.zeros((Ntest,16))
        erri = np.zeros((Ntest,16))
        maxErre = np.zeros(Ntest)
        maxErri = np.zeros(Ntest)
        eNorm = 0.
        iNorm = 0.
        xtest = np.zeros(2); vtest = np.zeros(2)

        for i in range(0,Ntest):
            xtest[0] = np.random.uniform(a[0],b[0],1)
            xtest[1] = np.random.uniform(a[1],b[1],1)
            vtest[0] = np.random.uniform(a[2],b[2],1)
            vtest[1] = np.random.uniform(a[3],b[3],1)

            Fe, Fi = self.computeEDF(xtest,vtest)
            FeMan = self.computeCDF(xtest,vtest,self.time,self.manSolne)
            FiMan = self.computeCDF(xtest,vtest,self.time,self.manSolni)
            erre[i,:] = FeMan - Fe
            erri[i,:] = FiMan - Fi
            eNorm = eNorm + np.max(np.abs(erre[i,:]))**2
            iNorm = iNorm + np.max(np.abs(erri[i,:]))**2
            maxErri[i] = np.max(np.abs(erri[i,:]))
            maxErre[i] = np.max(np.abs(erre[i,:]))
            if maxErre[i] > 1.e-1:
                print(maxErre[i])
                print(Fe)
                print(FeMan)
                print(erre[i,:])
                print((xtest,vtest))
            print(i)
        eNorm = np.sqrt(eNorm)
        iNorm = np.sqrt(iNorm)
        infNormi = np.max(maxErri)
        infNorme = np.max(maxErre)
        print('eNormInf')
        print(infNorme)
        print('iNormInf')
        print(infNormi)
        print('eNOrm')
        print(eNorm)
        print('iNorm')
        print(iNorm)

        
    def InterpGridHydros(self,i):
        if self.varwts:
            self.rho[i] = self.IG(self.parwtse, self.parwtsi)
        else: 
            self.rho[i] = self.IG(np.ones(self.x.shape[0]), np.ones(self.x.shape[0]))
        #self.u[i] = self.IGVec(self.v)
        #self.energy[i] = self.IG(self.v[:,0]**2 + self.v[:,1]**2)
        self.kenergy[i] = 0.5*np.sum(self.ve[:,0]**2 + self.ve[:,1]**2)/self.xe.shape[0]
        self.kenergy[i] += 0.5*np.sum(self.vi[:,0]**2 + self.vi[:,1]**2)/self.xi.shape[0]

        
    def Initialize(self,N):
        self.xi, self.vi, self.parwtsi, self.f0i, self.xe, self.ve, self.parwtsi, self.f0e = self.IDatGen(N,self.DomLenX,self.DomLenY, 5.)
        self.InterpGridHydros(0)
#        self.phi[0] = SO.Poisson2Dperiodic(self.chrge*(self.rho[0] - np.mean(self.rho[0]))*self.DomLenX*self.DomLenY,self.DomLenX,self.DomLenY)
        self.phi[0] = SO.Poisson2Dperiodic(self.rho[0],self.DomLenX,self.DomLenY)
        self.E[0,:,:,1], self.E[0,:,:,0] = SO.SpectralDerivative2D(-1.*self.phi[0],self.DomLenX,self.DomLenY)
        
        ## Do an initial backward half-step to offset positions and velocities in time
        self.Epare, self.Epari = self.IPVec(self.E[0])
        dt_tmp = -1.*0.5*self.dt

        self.vi += 0.5*dt_tmp*self.chrgi*self.Epari
        t = self.chrgi*self.B*dt_tmp/2.0; s = 2.0*t/(1.0 + t*t); c = (1-t*t)/(1+t*t)
        vnewx = self.vi[:,0]*c - self.vi[:,1]*s
        vnewy = self.vi[:,0]*s + self.vi[:,1]*c
        self.vi[:,0] = vnewx; self.vi[:,1] = vnewy
        self.vi += 0.5*dt_tmp*self.chrgi*self.Epari
        
        self.ve += 0.5*dt_tmp*self.chrge*self.Epare
        t = self.chrge*self.B*dt_tmp/2.0; s = 2.0*t/(1.0 + t*t); c = (1-t*t)/(1+t*t)
        vnewx = self.ve[:,0]*c - self.ve[:,1]*s
        vnewy = self.ve[:,0]*s + self.ve[:,1]*c
        self.ve[:,0] = vnewx; self.ve[:,1] = vnewy
        self.ve += 0.5*dt_tmp*self.chrge*self.Epare

    def ComputeFieldPoisson(self,i):
#        self.phi[i] = SO.Poisson2Dperiodic(self.chrge*(self.rho[i] - np.mean(self.rho[i]))*self.DomLenX*self.DomLenY,self.DomLenX,self.DomLenY)
        self.phi[i] = SO.Poisson2Dperiodic(self.rho[i],self.DomLenX,self.DomLenY)
        self.E[i,:,:,1], self.E[i,:,:,0] = SO.SpectralDerivative2D(-1.*self.phi[i],self.DomLenX,self.DomLenY)

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
                print('parwtse[0] = ' + str(self.parwtse[0]))
            self.TestEField(i)
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
