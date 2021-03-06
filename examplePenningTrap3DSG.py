# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:38:41 2016

@author: ricketsonl
"""
import numpy as np
import PICES_3DSG as PIC3D
import time
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy.special import ellipk, ellipe

## User Input ##
dsizex = 160.; dsizey = 160.; dsizez = 160.
ncx = 128
T = 15.
step_per_tp = 10.
N_per_sparse_cell = 10

n = int(np.log2(ncx))

## Derived Data ##
nstep = int(step_per_tp*T)
N = int((4*ncx*(n+1)*n/2. + 2*ncx*n*(n-1)/2. + ncx*(n-1)*(n-2)/2.)*N_per_sparse_cell)
delx = dsizex/ncx; delt = 1./step_per_tp
CFL = delx/delt

ncel_pic = 128

print('Total particle number: ' + str(N))
print('Proximity to CFL limit (stable if > 1): ' + str(CFL))

def idatgen(N,Lx,Ly,Lz):
    xx = np.random.normal(Lx/2.,0.18*Lx,N) % Lx
    xy = np.random.normal(Ly/2.,0.05*Ly,N) % Ly
    xz = np.random.normal(Lz/2.,0.25*Lz,N) % Lz
    x = np.column_stack((xx,xy,xz))
    
    vx = np.random.normal(0.,1.,N); vy = np.random.normal(0.,1.,N); vz = np.random.normal(0.,1.0,N)
    v = np.column_stack((vx,vy,vz))
    
    return x, v

def Bfield(x):
    B = np.zeros(x.shape)
    B[:,2] = 5.
    return B
    
V0 = 60.*dsizez
def Efield_ext(x):
    E = np.zeros(x.shape)
    E[:,2] = (x[:,2] - dsizez/2.)*V0/(dsizez*dsizez)
    E[:,0] = -(x[:,0] - dsizex/2.)*V0/(2.*dsizez*dsizez)
    E[:,1] = -(x[:,1] - dsizey/2.)*V0/(2.*dsizez*dsizez)
    return E

bstart = time.time()
schimp = PIC3D.PICES3DSG(T,dsizex,dsizey,dsizez,ncx,nstep,idatgen,Bfield,N,ifEext=True,Eext=Efield_ext,rho_overall=0.1)
bend = time.time()
print('Time to build scheme object: ' + str(bend-bstart))

start = time.time()
schimp.Run()
end= time.time()

timp = end - start

KE = schimp.KE; PE = schimp.PEnergy()
t = np.linspace(0.,schimp.T,num=KE.shape[0])

print('Computed energies')

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

print('Created plots')


delE = np.amax(KE+PE)/np.amin(KE+PE) - 1.

print('Run took '+str(timp)+' seconds.')
print('Fractional energy change: '+str(delE))


rgs = time.time()
#rhoi = schimp.rho[0].RegularGrid(ncel_pic,ncel_pic,ncel_pic)
rhof = schimp.rho[-1].RegularGrid(ncel_pic,ncel_pic,ncel_pic)
rge = time.time()

print('Time to generate regular grids: ' + str(rge-rgs))
    
rmax = np.amax(rhof); rmin = np.amin(rhof)

x = np.linspace(0.,dsizex,ncel_pic,endpoint=False)
y = np.linspace(0.,dsizey,ncel_pic,endpoint=False)
z = np.linspace(0.,dsizez,ncel_pic,endpoint=False)

Y, X, Z = np.meshgrid(x,y,z)


## Generate 3D contour plot of final density with Mayavi ##
mlab.figure(bgcolor=(1,1,1),fgcolor=(0,0,0),size=(1024,768))
mlab.contour3d(X,Y,Z,rhof/np.mean(rhof),contours=32,opacity=.5,colormap='jet',vmax=np.amax(rhof)/np.mean(rhof),vmin=np.amin(rhof)/np.mean(rhof))
mlab.colorbar(orientation='vertical')
mlab.view(azimuth=30,elevation=60,distance='auto')
mlab.axes()
mlab.savefig('MayaviFrames/Cyclotron3D_nc' + str(ncx) + '_nppc' + str(N_per_sparse_cell) + '_final.png',size=(640,480))

mlab.show()

plt.show()

