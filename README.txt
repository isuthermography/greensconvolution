greensconvolution is a fast GPU-based implementation
of some thermal Green's function calculations and
surrogates for Green's function convolutions that
are needed by the "greensinversion" model-based
inversion for flash thermography.

Requirements
------------
Python -- Tested with Python 2.7; should work with Python 3.x
          but might need minor compatibility bugfixes
Numpy  -- Any recent version should be fine
Scipy  -- Any recent version should be fine
Cython -- Any recent version should be fine. Cython will need to be
          configured with a suitable C compiler. On Linux this is
	  usually handled by your package manager. On Windows, see
	  https://github.com/cython/cython/wiki/installingonwindows and
	  https://github.com/cython/cython/wiki/CythonExtensionsOnWindows
OpenCL -- You will also to have the OpenCL installable client driver
	  available. On Linux this is usually as simple as 
	  "dnf install ocd-icd-devel". On Windows make sure you have
	  the OpenCL drivers provided by your GPU card vendor installed
PyOpenCL -- From https://mathema.tician.de/software/pyopencl/
          On Linux this may be available with your package manager,
	  e.g. "dnf install python2-pyopencl".
NetCDF4 Python bindings -- http://unidata.github.io/netcdf4-python/. On
                           Linux this may be as simple as
			   "dnf install python2-netcdf4"
LaTeX  -- Needed if you want to build the greensfcn_doc.pdf mathematical
          documentation

INSTALLATION

To build greensconvolution:
   python setup.py build
To install into site-packages (may need to be root or Administrator)
   python setup.py install

The math behind greensconvolution (the flat case anyway)
is documented in greensfcn_doc.pdf. To build this from
greensfcn_doc.tex, make sure LaTeX is installed, then run:
   pdflatex greensfcn_doc.te

VERIFYING CORRECT OPERATION

Run the demos/verification.py script:
  cd demos
  python verification.py
Check for any large error percentages. All except the
"amplitude factor approximation error" should be significantly
less than 1%. The amplitude factor approximation errors
should be around 1.5% or less.
Example output from verification.py:
  Flat Direct: 4.691705
  Flat GC quadpack: 4.691703
  Flat GC interpolator: 4.690710
  Flat GC quadpack error: -0.000030%
  Flat GC interpolator error: -0.021211%
  amplitude factor approximation error (convex): 1.487607%
  amplitude factor approximation error (concave): -1.160656%
  Concave GC interpolator error: -0.016212%
  Convex GC interpolator error: -0.028710%
  Image source flat = 0.099146
  Image source flat GC = 0.099146
  Image source flat error = -0.000030%
  image source concave error = -0.000034%
  concave_gf_error = 0.000006%
  convex_gf_error = -0.000009%
  concave_gf_cl_error = 0.000000%
  convex_gf_cl_error = 0.000000%


REBUILDING THE greensconvolution.nc CACHE
(this step should not be necessary as the
included copy should be fine):
   * Change to the source directory, i.e. greensconvolution/greensconvolution
   * Run greensconvolution_calc.py as a script, e.g
     python -i greensconvolution_calc.py
   * This script will write a new "greensconvolution.nc" into /tmp
   * The new "greensconvolution.nc" should be tested with the
     "greensconvolution_test.py" script in the demos/ directory.
   * Once validated it can replace the "greensconvolution.nc" in the
     source directory. 


Greensinversion is a package for model-based inversion of
flash thermography measurement data.


ACKNOWLEDGMENTS

If using or building on this software please cite the authors!
S.D. Holland and B. Schiefelbein, Model-based inversion for pulse thermography, J. Exp. Mech (under review, 2018)

This material is based own work supported by NASA through Early
Stage Innovation grant #NNX15AD75G.

Copyright (C) 2015-2018 Iowa State University Research Foundation, Inc.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

