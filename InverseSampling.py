# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:00:19 2016

@author: lfr
"""

import numpy as np
from numpy.random import uniform

def NewtonRoot(F,Fprime,init,tol=1.e-5,maxits=100):
    x = np.asarray(init)
    res = np.amax(np.abs(F(x)))
    i = 0
    while res > tol and i < maxits:
        v = F(x); dv = Fprime(x)
        x = x - v/dv
        res = np.amax(np.abs(F(x)))
        i += 1
    if np.amax(res) > tol:
        print('Failed to converge with residual = ' + str(res))
    #else:
        #print('Maximum residual: ' + str(res))
    
    return x
    
def InverseSampler(F,Fprime,N,L,t=1.e-5,mits=100):
    u = uniform(low=0.,high=1.,size=N)
    def diff_Func(x):
        return F(x) - u

    sol = NewtonRoot(diff_Func,Fprime,u,tol=t,maxits=mits)
    
    return sol
    