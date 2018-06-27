import sys
from Cython.Build import cythonize
from numpy.distutils.core import setup as numpy_setup, Extension as numpy_Extension

ext_modules=cythonize("greensconvolution/*.pyx")

emdict=dict([ (module.name,module) for module in ext_modules])

gcf_pyx_ext=emdict['greensconvolution.greensconvolution_fast']
gcf_pyx_ext.sources.append("greensconvolution/greensconvolution_fast_c.c")
#gcf_pyx_ext.extra_compile_args=['-g']
gcf_pyx_ext.extra_compile_args=['-fopenmp','-O3']
gcf_pyx_ext.extra_link_args=['-lgomp']


numpy_setup(name="greensconvolution",
            description="greensconvolution",
            author="Stephen D. Holland",
            url="http://thermal.cnde.iastate.edu",
            ext_modules=ext_modules,
            packages=["greensconvolution"],
            package_data={ "greensconvolution":
                           [ "greensconvolution.nc","greensconvolution_fast_c.c","opencl_interpolator_prefix.c","simplegaussquad.c","quadpack_prefix.c","qagse_fparams.c","quadpack.c","imagesources.c","imagesources_curved.c","greensfcn_curved.c" ]  # Note: greensconvolution.nc file can be regenerated in /tmp by running greensconvolution/greensconvolution_calc.py as a script -- preferably in ipython
            })
