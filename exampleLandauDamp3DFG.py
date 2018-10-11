# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:38:41 2016

@author: ricketsonl
"""
import numpy as np
import PICES3D_Exp_Clean_LowMem as PIC3D
import time
import matplotlib.pyplot as plt
import InverseSampling as IS

## User Input ##
dsizex = 160.; dsizey = 160.; dsizez = 160.
#dsizex = 2.*np.pi; dsizey = 2.*np.pi
#dsizex = 21.; dsizey = 21.
ncx = 128
T = 2.7
step_per_tp = 20.
N_per_cell = 1


## Derived Data ##
nstep = int(step_per_tp*T)
N = int(ncx**3*N_per_cell)
delx = dsizex/ncx; delt = 1./step_per_tp
CFL = delx/delt

print('Total particle number: ' + str(N))
print('Proximity to CFL limit (stable if > 1): ' + str(CFL))

def idatgen(N,Lx,Ly,Lz):
    #xx = np.random.normal(0.5*Lx,0.07*Lx,N) % Lx; xy = np.random.normal(0.5*Ly,0.2*Ly,N) % Ly #<--- Cyclotron
    #x = np.column_stack((xx,xy))

    Fy = lambda r: r/Ly + 0.2*np.sin(8.*np.pi*r/Ly)/(8.*np.pi)
    Fyprime = lambda r: (1. + 0.2*np.cos(8.*np.pi*r/Ly))/Ly
    Fx = lambda r: r/Lx + 0.15*np.sin(6.*np.pi*r/Lx)/(6.*np.pi)
    Fxprime = lambda r: (1. + 0.15*np.cos(6.*np.pi*r/Lx))/Lx
    xx = IS.InverseSampler(Fx,Fxprime,N,Lx,t=1.e-7) % Lx; xy = IS.InverseSampler(Fy,Fyprime,N,Ly,t=1.e-7) % Ly
    xz = IS.InverseSampler(Fx,Fxprime,N,Lx,t=1.e-7) % Lz
    x = np.column_stack((xx,xy,xz))
    
    vx = np.random.normal(0.,1.,N); vy = np.random.normal(0.,1.,N); vz = np.random.normal(0.,1.,N)
    v = np.column_stack((vx,vy,vz))
    
    return x, v
    
def Bfield(x):
    return np.zeros((x.shape[0],3))
    

schimp = PIC3D.PICES3D_Exp_Clean(T,dsizex,dsizey,dsizez,ncx,ncx,ncx,nstep,idatgen,Bfield,N)
start = time.time()
schimp.Run()
end= time.time()

timp = end - start

KE = schimp.KE; PE = schimp.PE
t = np.linspace(0.,schimp.T,num=KE.shape[0])
plt.figure(2)
plt.plot(t,KE,'b',t,PE,'r',t,PE+KE,'g',lw=2)
plt.title('Energy Conservation')

k = 2.*np.pi/dsizex
w = np.sqrt(1. + 3.*k**2)
vph = w/k
gamma = 0.5*np.sqrt(np.pi/2.)*(vph)**3*(1./w)**2*np.e**(-vph**2/2.)

plt.figure(3)
plt.semilogy(t,PE,'b',t,0.6*PE[0]*np.exp(-gamma*t),'k',lw=2)
plt.title('Potential Energy')

plt.show()

delE = np.amax(KE+PE)/np.amin(KE+PE) - 1.

print('Took '+str(timp)+' seconds.')
print('Fractional energy change: '+str(delE))

########################################################################
# Mayavi fiddling #
########################################################################
from mayavi import mlab

rmax = np.amax(schimp.rho); rmin = np.amin(schimp.rho)
#nsteps = schimp.rho.shape[0]

mlab.figure(bgcolor=(1,1,1),fgcolor=(0,0,0),size=(640,480))
plot = mlab.contour3d(schimp.rho/np.mean(schimp.rho),contours=8,opacity=.85,colormap='jet')
mlab.outline(color=(0,0,0),line_width=2)
mlab.colorbar(orientation='vertical')
mlab.view(azimuth=30,elevation=60,distance='auto')
mlab.savefig('MayaviFrames/FGLandau_nc' + str(ncx) + '_np' + str(int(N/1e6)) + 'e6.png',size=(640,480))
"""
for i in range(1,nsteps):
    fullname = 'MayaviFrames/FGLandau%03d.png' %i
    plot.mlab_source.scalars = schimp.rho[i]
    mlab.savefig(fullname,size=(640,480))"""
    
mlab.show()