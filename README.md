# SparsePIC ReadMe #

SparsePIC is a particle-in-cell (PIC) code that uses the sparse grid
combination technique for variance reduction and accelerating the field 
solve(s).  At present the code is electrostatic (externally applied magnetic 
fields are supported), and supports only periodic boundary conditions on 
rectangular domains.  

1. Getting up and running:
The high-level code is written in Python with calls to C routines generated 
using Cython.  You'll need the Anaconda python distribution (or similar... 
mainly you need numpy, scipy, and matplotlib), version 2.7.x, and a C 
compiler (tested with gcc).  

Additionally, for the 3-D visualizations that are implemented in the examples, 
you'll need Mayavi.  If you have Anaconda, getting Mayavi is as simple as 
'conda install mayavi' from the command line.  

You can specify your compiler for the C routines in the file 'setup.py'.  You 
should edit line 13 in that file to read

os.environ["CC"] = "your-compiler-name-here"

Then, you can build everything with:

python setup.py build_ext --inplace

2. Examples:
Once everything is built, you should be able to execute any of the 'example*.py' 
files in the directory.  The parameters currently in each file are set up so that 
it takes no more than 2 minutes to run on my laptop.  

Comments in the example files will give you a base-line understanding of how to run 
the code.  More documentation to come soon.  

exampleCyclotron2DSG.py: A 2-D example with sparse grids in a uniform magnetic field 
out of the plane.  Approximates the dynamics of a coasting beam in a cyclotron, 
displaying the expected 'spiral galaxy'-like dynamics due to the ExB drift.

exampleDiocotronSG.py: 2-D example with sparse grids of a ring of electrons on a 
uniform, fixed background of ions.  ExB drift has a shearing effect that leads to 
vortex formation.

exampleDiocotronFG.py: Same as above, but with standard, full grid PIC instead of 
sparse grids.  

exampleLandauDamp3DFG.py: Non-linear Landau damping in 3-D periodic box.  Uses 
standard, full grids.

examplePenningTrap3DSG.py: 3-D simulation of a Penning Trap-like setup.  The same 
spiral galaxy behavior from the cyclotron example is seen in the x-y plane, along 
with oscillatory dynamics in z due to quadrupole electric field. 
