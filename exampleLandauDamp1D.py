# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:38:41 2016

@author: ricketsonl
"""
import numpy as np
import PICES_1D_multispecies as PIC1D
import time
import matplotlib.pyplot as plt
import InverseSampling as IS

## User Input ##
dsizex = 160.
ncx = 50
T = 300.
step_per_tp = 5.
N_per_cell = 5000


## Derived Data ##
nstep = int(step_per_tp*T)
N = int(ncx*N_per_cell)
delx = dsizex/ncx; delt = 1./step_per_tp
CFL = delx/delt

print('Total particle number: ' + str(N))
print('Proximity to CFL limit (stable if > 1): ' + str(CFL))

def idatgen(N,Lx):
    #xx = np.random.normal(0.5*Lx,0.07*Lx,N) % Lx; xy = np.random.normal(0.5*Ly,0.2*Ly,N) % Ly #<--- Cyclotron
    #x = np.column_stack((xx,xy))

    Fx = lambda r: r/Lx + 0.25*np.sin(2.*np.pi*r/Lx)/(2.*np.pi)
    Fxprime = lambda r: (1. + 0.25*np.cos(2.*np.pi*r/Lx))/Lx
    xi = IS.InverseSampler(Fx,Fxprime,N,Lx,t=1.e-7) % Lx
    
    vi = np.random.normal(0.,1.,N)
    
    xe = np.random.uniform(0.,Lx,N)
    ve = np.random.normal(0.,1.,N)

    return xi, vi, xe, ve
    
def Bfield(x):
    return np.zeros((x.shape[0],3))
    

schimp = PIC1D.PICES1D(T,dsizex,ncx,nstep,idatgen,N,varwts=False)
start = time.time()
schimp.Run(N)
end= time.time()

timp = end - start

KE = schimp.KEnergy(); PE = schimp.PEnergy()
t = np.linspace(0.,schimp.T,num=KE.shape[0])
plt.figure(2)
plt.plot(t,KE,'b',t,PE,'r',t,PE+KE,'g',lw=2)
plt.title('Energy Conservation')

k = 2.*np.pi/dsizex
w = np.sqrt(1. + 3.*k**2)
vph = w/k
gamma = 0.5*np.sqrt(np.pi/2.)*(vph)**3*(1./w)**2*np.e**(-vph**2/2.)

plt.figure(3)
plt.semilogy(t,PE,'b',lw=2)
plt.title('Potential Energy')

print(KE+PE)
delE = np.amax(KE+PE)/np.amin(KE+PE) - 1.

print('Took '+str(timp)+' seconds.')
print('Fractional energy change: '+str(delE))

schimp.AnimateResult('test_anim',8,6,4.,16.,0.)

plt.show()
