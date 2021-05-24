import os
import os.path
import subprocess
import re
from setuptools import setup
from setuptools.command.install_lib import install_lib
from setuptools.command.install import install
from setuptools.command.build_ext import build_ext
import setuptools.command.bdist_egg
import sys
import distutils.spawn
import numpy as np
from Cython.Build import cythonize

extra_compile_args = {
    "msvc": ["/openmp","/D_USE_MATH_DEFINES=1"],
    #"unix": ["-O0", "-g", "-Wno-uninitialized"),    # Replace the line below with this line to enable debugging of the compiled extension
    "unix": ["-fopenmp","-O5","-Wno-uninitialized"],
    "clang": ["-fopenmp","-O5","-Wno-uninitialized"],
}

extra_include_dirs = {
    "msvc": [".", np.get_include() ],
    "unix": [".", np.get_include() ],
    "clang": [".", np.get_include() ],
}

extra_libraries = {
    "msvc": [],
    "unix": ["gomp",],
    "clang": [],
}

extra_link_args = {
    "msvc": [],
    "unix": [],
    "clang": ["-fopenmp=libomp"],
}


class build_ext_compile_args(build_ext):
    def build_extensions(self):
        compiler=self.compiler.compiler_type
        for ext in self.extensions:
            if compiler in extra_compile_args:
                ext.extra_compile_args=extra_compile_args[compiler]
                ext.extra_link_args=extra_link_args[compiler]
                ext.include_dirs.extend(list(extra_include_dirs[compiler]))
                ext.libraries.extend(list(extra_libraries[compiler]))
                pass
            else:
                # use unix parameters as default
                ext.extra_compile_args=extra_compile_args["unix"]
                ext.extra_link_args=extra_link_args["unix"]
                ext.include_dirs.extend(list(extra_include_dirs["unix"]))
                ext.libraries.extend(extra_libraries["unix"])
                pass
                
            pass
            
        
        build_ext.build_extensions(self)
        pass
    pass



ext_modules=cythonize("greensconvolution/*.pyx")

emdict=dict([ (module.name,module) for module in ext_modules])

gcf_pyx_ext=emdict['greensconvolution.greensconvolution_fast']
gcf_pyx_ext.sources.append("greensconvolution/greensconvolution_fast_c.c")
#gcf_pyx_ext.extra_compile_args=['-g']
gcf_pyx_ext.extra_compile_args=['-fopenmp','-O3']
gcf_pyx_ext.extra_link_args=['-lgomp']


setup(name="greensconvolution",
      description="greensconvolution",
      author="Stephen D. Holland",
      url="http://thermal.cnde.iastate.edu",
      ext_modules=ext_modules,
      packages=["greensconvolution"],
      cmdclass = {
          "build_ext": build_ext_compile_args,
      },
      package_data={ "greensconvolution":
                     [ "greensconvolution.nc","greensconvolution_fast_c.c","opencl_interpolator_prefix.c","simplegaussquad.c","quadpack_prefix.c","qagse_fparams.c","quadpack.c","imagesources.c","imagesources_curved.c","greensfcn_curved.c" ]  # Note: greensconvolution.nc file can be regenerated in /tmp by running greensconvolution/greensconvolution_calc.py as a script -- preferably in ipython
                 })
