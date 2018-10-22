# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 20:51:22 2016

@author: ricketsonl
"""
import numpy as np
import PICES_2DFG as PIC2DFG
import time
import matplotlib.pyplot as plt
import InverseSampling as IS

## User Input ##
dsizex = 40.; dsizey = 40.
ncx = 128; ncy = 128
T = 30.
step_per_tp = 10.
N_per_cell = 60

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
def idatgen(N,Lx,Ly):
    ### Ring ###
    R = np.random.normal(0.5*(Lx+Ly)/4.,0.025*Lx,N); theta = np.random.uniform(0.,2.*np.pi,N)
    xx = (R*np.cos(theta) + 0.5*Lx) % Lx
    xy = (R*np.sin(theta) + 0.5*Ly) % Ly
    x = np.column_stack((xx,xy))
    
    vx = np.random.normal(0.,1.,N); vy = np.random.normal(0.,1.,N)
    v = np.column_stack((vx,vy))

    return x, v

## The forcing function on the right side of the Vlasov equation.  Just a sine for demsonstration here. ##
def RHSfunc(x,v,t):
    N = x.shape[0]
    return 0.5*np.sin(t)*np.ones(N)

## Build PIC object ##
schimp = PIC2DFG.PICES2DFG(T,dsizex,dsizey,ncx,ncy,nstep,idatgen,N,varwts=True,RHS=RHSfunc)
## Set magnetic field ##
schimp.B = 15.

## Run scheme and time it ##
start = time.time()
schimp.Run(N)
end= time.time()

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
