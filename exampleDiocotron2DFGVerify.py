# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 20:51:22 2016

@author: ricketsonl
"""
import numpy as np
import PICES_2DFG_Verify as PIC2DFG
import time
import matplotlib.pyplot as plt
import matplotlib
import InverseSampling as IS

import cProfile
import pstats

## User Input ##
dsizex = 10.; dsizey = 10.
ncx = 64; ncy = 64
T = 2.
step_per_tp = 32
N_per_cell = 4*4*88

print_prof = True

## Derived Data ##
nstep = int(step_per_tp*T) # total number of time-steps
N = int(ncx*ncy*N_per_cell) # total number of particles

print('Total particle number = ' + str(N))

# Compute and print CFL number for reference
delx = dsizex/ncx; delt = 1./step_per_tp
CFL = delx/delt
print('Proximity to CFL limit (stable if > 1): ' + str(CFL))

## Name of movie showing density evolution to be saved ##
anim_name = 'DiocotronForcedFG'

## Function that generates initial particle positions and velocities ##
## Both x and v should have shape (N,2) ##
def idatgen(N,Lx,Ly,MAX):

    xxe = np.zeros(N); xxi = np.zeros(N); xye = np.zeros(N); xyi = np.zeros(N); wi = np.zeros(N); f0i = np.zeros(N)
    vxe = np.zeros(N); vxi = np.zeros(N); vye = np.zeros(N); vyi = np.zeros(N); we = np.zeros(N); f0e = np.zeros(N)

    #for i in range(0,N):
    xxe = np.random.uniform(0.,Lx,size=N)
    xxi = np.random.uniform(0.,Lx,size=N)
#        vxe[i] = np.random.uniform(-0.5*Lx,0.5*Lx,1)
#        vxi[i] = np.random.uniform(-0.5*Lx,0.5*Lx,1)
    vxe = np.random.normal(0.,1.,size=N)
    vxi = np.random.normal(0.,1.,size=N)
    xye = np.random.uniform(0.,Ly,size=N)
    xyi = np.random.uniform(0.,Ly,size=N)
#        vye[i] = np.random.uniform(-0.5*Ly,0.5*Ly,1)
#        vyi[i] = np.random.uniform(-0.5*Ly,0.5*Ly,1)
    vye = np.random.normal(0.,1.,size=N)
    vyi = np.random.normal(0.,1.,size=N)
    f0e = 1./(Lx*Ly*2*np.pi)*np.exp(-0.5*vxe**2-0.5*vye**2)
    f0i = 1./(Lx*Ly*2*np.pi)*np.exp(-0.5*vxi**2-0.5*vyi**2)
        #we[i]  = manufacturedSolne((xxe[i],xye[i]), (vxe[i],vye[i]), 0., Lx, Ly)/f0e[i]
        #wi[i]  = manufacturedSolni((xxi[i],xyi[i]), (vxi[i],vyi[i]), 0., Lx, Ly)/f0i[i]

    xe = np.column_stack((xxe,xye))
    ve = np.column_stack((vxe,vye))
    xi = np.column_stack((xxi,xyi))
    vi = np.column_stack((vxi,vyi))

    we = manufacturedSolne(xe, ve, 0., Lx, Ly)/f0e
    wi = manufacturedSolne(xi, vi, 0., Lx, Ly)/f0i
#    print('we[0] = ' + str(we[0]))
#    print('f0e[0] = ' + str(f0e[0]))
#    print('xi[0]')
#    print(xi[0])

    return xi, vi, wi, f0i, xe, ve, we, f0e

## The manufactured distribution function corresponding to particles with positive charge.
## Computed in the continuum not on the grid.  Fi: R^2 X R^2 X R^1 --> R^1
def manufacturedSolni(x,v,t,Lx,Ly):
# Fi = 4*M/np.pi*u^2*v^2*np.exp(-u^2-v^2)
    M = np.maximum(4.*np.pi**2/Lx**2, 4.*np.pi**2/Ly**2)
    P = 1./(M*Lx*Ly)

    Fi = 4.*P*M/np.pi*v[0]**2*v[1]**2*np.exp(-v[0]**2-v[1]**2)
    
    return Fi

def testSoln(x,v,t,Lx,Ly):
    return 1.
## The manufactured distribution function corresponding to particles with negative charge.
## Computed in the continuum not on the grid Fe: R^2 X R^2 X R^1 --> R^1
def manufacturedSolne(x,v,t,Lx,Ly):
# Fe = 4/np.pi*u^2*v^2*np.exp(-u^2-v^2)*(M-4*np.pi^2*np.sin(np.pi*t)*(np.sin(2*np.pi*x/Lx)/Lx^2 + np.cos(2*np.pi*y/Ly)/Ly^2)).
    M = np.maximum(4.*np.pi**2/Lx**2, 4.*np.pi**2/Ly**2)
    P = 1./(M*Lx*Ly)
   
    fv = (4./np.pi)*(v[:,0]**2)*(v[:,1]**2)*np.exp(-v[:,0]**2-v[:,1]**2)
    fx = M*P - 4.*np.pi**2*P*np.sin(np.pi*t)*(np.sin(2.*np.pi*x[:,0]/Lx)/Lx**2 + np.cos(2.*np.pi*x[:,1]/Ly)/Ly**2)
   
#    Fe = 4./np.pi*v[0]**2*v[1]**2*np.exp(-v[0]**2-v[1]**2)*(M*P-4*np.pi**2*P*np.sin(np.pi*t)*(np.sin(2*np.pi*x[0]/Lx)/Lx**2 + np.cos(2*np.pi*x[1]/Ly)/Ly**2))
    Fe = fv*fx
    return Fe

## The electric field which is consistent with the above distribution functions.
## Computed in the continuum not on the grid.  Only depends on space, no velocity.  E: R^2 X R^1 --> R^2
def manufacturedEField(x,t,Lx,Ly):
# E = < -2*np.pi/Lx*np.sin(np.pi*t)*np.cos(2*np.pi*x/Lx), 2*np.pi/Ly*np.sin(np.pi*t)*np.sin(2*np.pi*y/Ly) >
    M = np.maximum(4.*np.pi**2/Lx**2, 4.*np.pi**2/Ly**2)
    P = 1./(M*Lx*Ly)

    E = np.zeros_like(x)
    if E.ndim > 1:
        E[:,0] = -2.*np.pi/Lx*P*np.sin(np.pi*t)*np.cos(2.*np.pi*x[:,0]/Lx)
        E[:,1] = 2.*np.pi/Ly*P*np.sin(np.pi*t)*np.sin(2.*np.pi*x[:,1]/Ly)
    else:
        E[0] = -2.*np.pi/Lx*P*np.sin(np.pi*t)*np.cos(2.*np.pi*x[0]/Lx)
        E[1] = 2.*np.pi/Ly*P*np.sin(np.pi*t)*np.sin(2.*np.pi*x[1]/Ly)
#    print('E = '+ str(E[:]))
    return E

def forcingFunctioni(x,v,t,qoverm,Lx,Ly):
    M = np.maximum(4.*np.pi**2/Lx**2, 4.*np.pi**2/Ly**2)
    P = 1./(M*Lx*Ly)
    Ft = 0
    Fx = 0
    Fy = 0
    Fu = M*P*(8./np.pi*v[:,1]**2*(v[:,0]-v[:,0]**3)*np.exp(-v[:,0]**2-v[:,1]**2))
    Fv = M*P*(8./np.pi*v[:,0]**2*(v[:,1]-v[:,1]**3)*np.exp(-v[:,0]**2-v[:,1]**2))
    Em = manufacturedEField(x,t,Lx,Ly)
    Si = Ft + v[:,0]*Fx+v[:,1]*Fy + qoverm*(Em[:,0]*Fu + Em[:,1]*Fv)
    return Si

def forcingFunctione(x,v,t,qoverm,Lx,Ly):
    M = np.maximum(4.*np.pi**2/Lx**2, 4.*np.pi**2/Ly**2)
    P = 1./(M*Lx*Ly)
    fv = 4./np.pi*v[:,0]**2*v[:,1]**2*np.exp(-v[:,0]**2-v[:,1]**2)
    fx = M*P - 4.*np.pi**2*P*np.sin(np.pi*t)*(np.sin(2.*np.pi*x[:,0]/Lx)/Lx**2 + np.cos(2.*np.pi*x[:,1]/Ly)/Ly**2)

    Ft = fv*(-4.*np.pi**3*P*np.cos(np.pi*t)*(np.sin(2.*np.pi*x[:,0]/Lx)/Lx**2 + np.cos(2.*np.pi*x[:,1]/Ly)/Ly**2))
    Fx = fv*(-1.*P*np.sin(np.pi*t)*(8.*np.pi**3/Lx**3*np.cos(2.*np.pi*x[:,0]/Lx)))
    Fy = fv*(P*np.sin(np.pi*t)*(8*np.pi**3/Ly**3*np.sin(2.*np.pi*x[:,1]/Ly)))
    Fu = fx*(8./np.pi*v[:,1]**2*(v[:,0]-v[:,0]**3)*np.exp(-1.*v[:,0]**2-v[:,1]**2))
    Fv = fx*(8./np.pi*v[:,0]**2*(v[:,1]-v[:,1]**3)*np.exp(-1.*v[:,0]**2-v[:,1]**2))
    Em = manufacturedEField(x,t,Lx,Ly)
    Se = Ft + v[:,0]*Fx+v[:,1]*Fy + qoverm*(Em[:,0]*Fu + Em[:,1]*Fv)
    return Se
## The forcing function on the right side of the Vlasov equation.  Just a sine for demsonstration here. ##
def RHSfunc(x,v,t):
    N = x.shape[0]
    return 0.5*np.np.sin(t)*np.ones(N)

## Build PIC object ##
schimp = PIC2DFG.PICES2DFGV(T,dsizex,dsizey,ncx,ncy,nstep,idatgen,N,varwts=True,RHSe=forcingFunctione,RHSi=forcingFunctioni,Efield=manufacturedEField,mSe=manufacturedSolne,mSi=manufacturedSolni)
#schimp = PIC2DFG.PICES2DFGV(T,dsizex,dsizey,ncx,ncy,nstep,idatgen,N,varwts=True,RHSe=forcingFunctione,RHSi=forcingFunctioni,Efield=manufacturedEField,mSe=testSoln,mSi=testSoln)

## Set magnetic field ##
schimp.B = 15.

## Run scheme and time it ##
start = time.time()
cProfile.run('schimp.Run(N)','runStats')
#schimp.Run(N)
end= time.time()

if print_prof:
    profdata = pstats.Stats('runStats')
    profdata.sort_stats('tottime')
    profdata.print_stats(12)

timp = end - start

KE = schimp.KEnergy(); PE = schimp.PEnergy()
t = np.linspace(0.,schimp.T,num=KE.shape[0])

## Plot total energy as a function of time ##
plt.figure(2)
plt.plot(t,KE,'b',t,PE,'r',t,PE+KE,'g',lw=2)
plt.title('Energy Conservation')

## Plot potential energy as function of time (mainly useful for Landau damping verification) ##
plt.figure(3)
plt.semilogy(t,PE,'b',lw=2)
plt.title('Potential Energy')


delE = np.amax(KE+PE)/np.amin(KE+PE) - 1.

print('Took '+str(timp)+' seconds.')
print('Fractional energy change: '+str(delE))

## Generate and save an animation of the density rho ##
tanis = time.time()
schimp.AnimateResult(anim_name,8,6,4.,16.,0.,zmax=np.amax(schimp.rho[0]),zmin=np.amin(schimp.rho[0]))
tanie = time.time()
tani = tanie - tanis
print('Animation took ' + str(tani) + ' seconds.')

## Show last frame of animation (i.e. rho at last time-step) ##
plt.show()
