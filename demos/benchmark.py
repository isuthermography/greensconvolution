import numpy as np
import time

from greensconvolution.greensconvolution_fast import greensconvolution_integrate
from greensconvolution.greensconvolution_calc import read_greensconvolution

#gc_kernel="opencl_interpolator" # greensconvolution kernel to use .. laptop: 18 sec
#gc_kernel="openmp_interpolator" # laptop: 19 sec
gc_kernel="opencl_quadpack"   # laptop: 1 min 10 sec
#gc_kernel="opencl_simplegaussquad"   # Laptop: VERY slow


# define materials:
composite_k=.138 # W/m/deg K
composite_rho=1.57e3 # W/m/deg K
composite_c=730 # J/kg/deg K

zrange=np.arange(.1e-3,5e-3,1e-3,dtype='f')  # step was .02e-3
xrange=np.arange(-20e-3,20e-3,1e-3,dtype='f') # step was .1e-3

zrange=zrange.reshape(zrange.shape[0],1,1)
xrange=xrange.reshape(1,xrange.shape[0],1)

rrange=np.sqrt(xrange**2.0+zrange**2.0)

trange=np.arange(10e-3,10.0,1.0,dtype='f') # step was 10e-3
trange=trange.reshape(1,1,trange.shape[0])


greensconvolution_params=read_greensconvolution()

cpu_starttime=time.time()

cpu_result=greensconvolution_integrate(greensconvolution_params,zrange,rrange,trange,composite_k,composite_rho,composite_c,1.0,(),kernel="openmp_interpolator")

cpu_elapsed=time.time()-cpu_starttime


starttime=time.time()

result=greensconvolution_integrate(greensconvolution_params,zrange,rrange,trange,composite_k,composite_rho,composite_c,1.0,(),kernel=gc_kernel)

elapsed=time.time()-starttime

GPUrate=np.prod(result.shape)/elapsed
# For 24x22mm tiles, 0.5 mm point spacing
# 24*22 -> 2112 surface points. Assume 1000 frames -> 2,112,000
# matrix rows. by 1240 colums = 2.62 billion evals. 
# Each

# Laptop with Intel GPU gives compute time of ~9000 seconds
# spec'd at ~ 325 Gflops

computetime=(24.0*22.0/0.5/0.5)*1000.*1240./GPUrate
print("Needed Compute time=%f s" % (computetime))

print("Rate: %f/usec" % (GPUrate/1e6))

docompare=cpu_result > 1e-6
print("Max relative error=%f%%" % (np.max(np.abs((result[docompare]-cpu_result[docompare])/cpu_result[docompare]))*100.0))

