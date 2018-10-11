# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:50:18 2016

@author: ricketsonl
"""
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

os.environ["CC"] = "g++-6"

ext_modules1=[Extension("InterpolationRoutines_MV",
                       ["InterpolationRoutines_MV.pyx"],
                       libraries=["m"],
                       extra_compile_args = ["-fopenmp","-O3","-ffast-math"],
                       extra_link_args = ["-fopenmp"])]
                       
ext_modules3=[Extension("Rotation",
                       ["Rotation.pyx"],
                       libraries=["m"],
                       extra_compile_args = ["-fopenmp","-O3","-ffast-math"],
                       extra_link_args = ["-fopenmp"])]                       
                       
ext_modules2=[Extension("InterpolationRoutines",
                        ["InterpolationRoutines.pyx"],
                        libraries=["m"],
                        extra_compile_args = ["-O3","-ffast-math"])]      
                        
EffSparseModules=[Extension("SparseGridInterpRoutines",
                        ["SparseGridInterpRoutines.pyx"],
                        libraries=["m"],
                        extra_compile_args = ["-fopenmp","-O3","-ffast-math"],
                        extra_link_args = ["-fopenmp"])]
                       
setup(name="InterpolationRoutines_MV",
      cmdclass = {"build_ext": build_ext},
      ext_modules = ext_modules1,include_dirs=[np.get_include()],)
      
setup(name="Rotation",
      cmdclass = {"build_ext": build_ext},
      ext_modules = ext_modules3,include_dirs=[np.get_include()],)
      
setup(name="InterpolationRoutines_",
      cmdclass = {"build_ext": build_ext},
      ext_modules = ext_modules2,include_dirs=[np.get_include()],)
      
setup(name="SparseGridInterpRoutines",
      cmdclass = {"build_ext": build_ext},
      ext_modules = EffSparseModules,include_dirs=[np.get_include()],)
