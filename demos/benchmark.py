import numpy as np
import time

from greensconvolution.greensconvolution_fast import greensconvolution_integrate
from greensconvolution.greensconvolution_fast import greensconvolution_greensfcn_curved
from greensconvolution.greensconvolution_calc import read_greensconvolution

gc_kernel="opencl_interpolator" # greensconvolution kernel to use .. laptop: 18 sec
#gc_kernel="openmp_interpolator" # laptop: 19 sec
#gc_kernel="opencl_quadpack"   # laptop: 1 min 10 sec
#gc_kernel="opencl_simplegaussquad"   # Laptop: VERY slow


# define materials:
composite_k=.138 # W/m/deg K
composite_rho=1.57e3 # W/m/deg K
composite_c=730 # J/kg/deg K

zrange=np.arange(.1e-3,5e-3,.002e-3,dtype='f')  # step was .02e-3
xrange=np.arange(-20e-3,20e-3,.001e-3,dtype='f') # step was .1e-3

zrange=zrange.reshape(zrange.shape[0],1,1)
xrange=xrange.reshape(1,xrange.shape[0],1)

rrange=np.sqrt(xrange**2.0+zrange**2.0)

trange=np.arange(10e-3,10.0,1.0,dtype='f') # step was 10e-3
trange=trange.reshape(1,1,trange.shape[0])


greensconvolution_params=read_greensconvolution()
greensconvolution_params.get_opencl_context("GPU")

cpu_starttime=time.time()

cpu_result=greensconvolution_integrate(greensconvolution_params,zrange,rrange,trange,0.0,composite_k,composite_rho,composite_c,1.0,(),kernel="openmp_interpolator")

cpu_elapsed=time.time()-cpu_starttime


starttime=time.time()

result=greensconvolution_integrate(greensconvolution_params,zrange,rrange,trange,0.0,composite_k,composite_rho,composite_c,1.0,(),kernel=gc_kernel)
#result=greensconvolution_greensfcn_curved(greensconvolution_params,np.ones((1,1,1),dtype='f'),rrange,zrange,rrange/5e-3,trange,composite_k,composite_rho,composite_c,(0,2),(1.0/5e-3)*np.ones((1,1,1),dtype='f'),np.zeros((1,1,1),dtype='f'),np.ones((1,1,1),dtype='f'),np.ones((1,1,1),dtype='f'))

elapsed=time.time()-starttime


GPUrate=rrange.shape[0]*rrange.shape[1]*trange.shape[2]/elapsed
CPUrate=np.prod(cpu_result.shape)/cpu_elapsed
# For 24x22mm tiles, 0.5 mm point spacing
# 24*22 -> 2112 surface points. Assume 1000 frames -> 2,112,000
# matrix rows. by 1240 colums = 2.62 billion evals. 
# Each

# Laptop with Intel GPU gives compute time of ~9000 seconds
# spec'd at ~ 325 Gflops

computetime=(24.0*22.0/0.5/0.5)*1000.*1240./GPUrate
print("Needed Compute time=%f s" % (computetime))

print("Rate: %f/usec" % (GPUrate/1e6))
print("CPU Rate: %f/usec" % (CPUrate/1e6))

#docompare=cpu_result > 1e-3
#print("Max relative error=%f%%" % (np.max(np.abs((result[docompare]-cpu_result[docompare])/cpu_result[docompare]))*100.0))

