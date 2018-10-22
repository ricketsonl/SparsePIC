# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:38:41 2016

@author: ricketsonl
"""
import numpy as np
import PICES_2DSG as PIC2D
import time
import matplotlib.pyplot as plt


## User Input ##
dsizex = 40.; dsizey = 40.      ## Size of the domain - measured in DeBye lengths
ncx = 512                       ## Effective resolution in each direction of the sparse grid to be used (must be power of 2 for now)
T = 40.0                        ## Final time of the simulation
step_per_tp = 20.               ## Time steps per inverse plasma frequency
N_per_sparse_cell = 30          ## Number of simulation particles per sparse cell

Bfield = 15.                    ## Strengh of magnetic field, i.e. cyclotron frequency / plasma frequency

ncx_save = ncx                  ## Resolution to use when animating result

## Derived Data ##
nstep = int(step_per_tp*T)
N = int(N_per_sparse_cell*ncx*(3.*np.log2(ncx)-1))

## Name of movie file to be generated
animname = 'DiocotronSG' + str(ncx) + 'c_' + str(N_per_sparse_cell) + 'p_' + str(int(Bfield)) + 'B'

print('Total particle number: ' + str(N))

## Compute and print CFL data
delx = dsizex/ncx; delt = 1./step_per_tp
CFL = delx/delt
print('Proximity to CFL limit (stable if > 1): ' + str(CFL))

## A function generating initial particle positions and velocities ##
def idatgen(N,Lx,Ly):
    R = np.random.normal(0.5*(Lx+Ly)/4.,0.025*Lx,N); theta = np.random.uniform(0.,2.*np.pi,N)
    xx = (R*np.cos(theta) + 0.5*Lx) % Lx
    xy = (R*np.sin(theta) + 0.5*Ly) % Ly
    x = np.column_stack((xx,xy))
    
    vx = np.random.normal(0.,1.,N); vy = np.random.normal(0.,1.,N)
    v = np.column_stack((vx,vy))
    
    return x, v

scheme = PIC2D.PICES2DSG(T,dsizex,dsizey,ncx,nstep,idatgen,N) ## Create scheme object
scheme.B = Bfield                                                      ## Set magnetic field

## Runs scheme and times execution ##
start = time.time()
scheme.Run()
end= time.time()

run_time = end - start

## Total kinetic and potential energy at each time step
KE = scheme.KE; PE = scheme.PEnergy()

delE = np.amax(KE+PE)/np.amin(KE+PE) - 1.

## Print timing and energy conservation data ##
print('Took '+str(run_time)+' seconds.')
print('Fractional energy change: '+str(delE))

rho0 = scheme.rho[0].RegularGrid(min(ncx,256),min(ncx,256))

## Animate density as a function of time ## 
animstart = time.time()
scheme.AnimateResult(animname,8,6,min(ncx,256),min(ncx,256),4.,12.,0,zmax=np.amax(rho0),zmin=np.amin(rho0),filt=False)
animend = time.time()
tani = animend - animstart

print('Animation took '+str(tani)+' seconds.')

## Show the last frame of the animation
plt.show()
