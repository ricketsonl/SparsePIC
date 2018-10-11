## Spectral operators in 1D ##

from numpy import *

def Poisson1Dperiodic(f,L):
    N = len(f)
    k = fft.fft(f)
    thek = N*fft.fftfreq(N)*2*pi/L
    thek[0] = 1.

    solk = k/thek**2
    solk[0] = 0
    sol = real(fft.ifft(solk))
    return sol

def Poisson2Dperiodic(f,Lx,Ly):
    Ny = len(f[:,0])
    Nx = len(f[0,:])
    k = fft.fftn(f)
    kxtmp = Nx*fft.fftfreq(Nx)*2*pi/Lx
    kytmp = Ny*fft.fftfreq(Ny)*2*pi/Ly

    thekx,theky = meshgrid(kxtmp,kytmp)
    div = thekx**2 + theky**2
    div[0,0] = 1.
    solk = k/div
    solk[0,0] = 0
    sol = real(fft.ifftn(solk))
    return sol
    
def Poisson3Dperiodic(f,Lx,Ly,Lz):
    Nx = f.shape[0]; Ny = f.shape[1]; Nz = f.shape[2]
    k = fft.fftn(f)
    kxtmp = Nx*fft.fftfreq(Nx)*2*pi/Lx
    kytmp = Ny*fft.fftfreq(Ny)*2*pi/Ly
    kztmp = Nz*fft.fftfreq(Nz)*2*pi/Lz
    
    thekx,theky,thekz = meshgrid(kxtmp,kytmp,kztmp,indexing='ij')
    div = thekx**2 + theky**2 + thekz**2
    div[0,0,0] = 1.
    solk = k/div
    solk[0,0,0] = 0.
    sol = real(fft.ifftn(solk))
    return sol    

def SpectralDerivative(f,L):
    N = len(f)
    k = fft.fft(f)
    thek = N*fft.fftfreq(N)*2*pi/L

    solk = k*1j*thek
    sol = real(fft.ifft(solk))
    return transpose(sol)

def SpectralDerivative2D(f,Lx,Ly):
    Ny = len(f[:,0])
    Nx = len(f[0,:])
    k = fft.fftn(f)
    kxtmp = Nx*fft.fftfreq(Nx)*2*pi/Lx
    kytmp = Ny*fft.fftfreq(Ny)*2*pi/Ly
    thekx, theky = meshgrid(kxtmp,kytmp)

    dfx_k = thekx*1j*k
    dfy_k = theky*1j*k

    dfx = real(fft.ifftn(dfx_k))
    dfy = real(fft.ifftn(dfy_k))

    return dfx, dfy
    
def SpectralDerivative3D(f,Lx,Ly,Lz):
    Nx = f.shape[0]; Ny = f.shape[1]; Nz = f.shape[2]
    k = fft.fftn(f)
    #print(k.shape,f.shape)
    
    kxtmp = Nx*fft.fftfreq(Nx)*2*pi/Lx
    kytmp = Ny*fft.fftfreq(Ny)*2*pi/Ly
    kztmp = Nz*fft.fftfreq(Nz)*2*pi/Lz    
    thekx,theky,thekz = meshgrid(kxtmp,kytmp,kztmp,indexing='ij')
    
    #print(thekx.shape,theky.shape,thekz.shape)
    
    dfx_k = thekx*1j*k
    dfy_k = theky*1j*k
    dfz_k = thekz*1j*k
    
    dfx = real(fft.ifftn(dfx_k)); dfy = real(fft.ifftn(dfy_k)); dfz = real(fft.ifftn(dfz_k))
    
    return dfx, dfy, dfz