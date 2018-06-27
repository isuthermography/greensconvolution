import numpy as np
import scipy as sp
import matplotlib
import pylab as pl
import scipy.integrate
from scipy.integrate import quad
from numpy.polynomial.legendre import leggauss
import time


try:
    import pyopencl as cl
    pass
except:
    cl=None
    pass



oldintegrand=lambda u,v,c: u**((-3.0)/2.0)*(v-u)**((-3.0)/2.0)*np.exp((-c**2/(v-u))-1/u)

oldintegral=np.vectorize(lambda v,c: quad(lambda u: oldintegrand(u,v,c),0,v))

integrand = lambda x,v,a: x**(-1.5)*(1.0-x)**(-1.5)*np.exp(-(1.0+a*x)/(v*x*(1.0-x)))



def newintegral(v,c,degree,num_repeats,kernel,doplots=False):
    # degree==300 seems to give accurate results
    
    oldvals=oldintegral(v,c)
    
    a=c**2.0-1.0
    assert( (a >= 0.0).all())  # a cannot be negative for our approximations

    if len(v) > 0: # a vector/matrix... must be numpy
       one = np.ones(v.shape,dtype='d')
       pass
    else:
        one=1.0
        pass
    
    epsilon=.05/np.max(np.array((v,one,a),dtype='d'),axis=0)

    # integral from 0 to epsilon and 1-epsilon to 1 each evaluate as
    # 2.5 x 10^-10 or less which we consider negligible

    scipyvals = (1.0/v**2.0)*np.vectorize(lambda v,a,epsilon: quad(lambda x: integrand(x,v,a),epsilon,1.0-epsilon))(v,a,epsilon)   
    
    # change integration region from epsilon to 1-epsilon to -1 to 1
    leadingcoefficient = (1.0-epsilon - epsilon)/2.0
    slope = ((1.0-epsilon - epsilon)/2.0).reshape(1,*epsilon.shape)
    offset = 0.5 # (epsilon + 1.0-epsilon)/2.0
    
    (points,weights)=leggauss(degree)

    # Worst case behavior around a=00, v=50.0... corresponds to very early time, directly over reflector
    
    pointsrs=points.reshape(points.shape[0],*np.ones(len(epsilon.shape),dtype='i4'))
    vals = (1.0/v**2.0) * leadingcoefficient * np.tensordot(weights.astype('f'),np.vectorize(integrand)(slope.astype('f')*pointsrs.astype('f')+offset,v.astype('f'),a.astype('f')),axes=(0,0))

    err = np.abs(vals-scipyvals[0])
    relerr= err/np.abs(scipyvals[0])  # division by zero warning expected here

    
    print("Max error = %g" % (np.max(err)))
    print("Is relative error of %f%%" % (100.0*np.max(err)/np.abs(scipyvals[0].ravel()[np.argmax(err)])))

    
    gpuvals=None
    if cl is not None:
        ctx=None
        
        # First search for first GPU platform 
        platforms = cl.get_platforms()
        for platform in platforms:
            has_gpu=[bool(device.type & cl.device_type.GPU) for device in platform.get_devices()]
            if np.any(has_gpu):
                
                ctx = cl.Context(
                    dev_type=cl.device_type.GPU,
                    properties=[(cl.context_properties.PLATFORM, platform)])
                pass
            pass
        
        if ctx is None:
            # fall back to a CPU platform 
            for platform in platforms:
                has_cpu=[bool(device.type & cl.device_type.CPU) for device in platform.get_devices()]
                if np.any(has_cpu):
                    
                    ctx = cl.Context(
                        dev_type=cl.device_type.CPU,
                        properties=[(cl.context_properties.PLATFORM, platform)])
                    pass
                pass
            pass
        
        queue=cl.CommandQueue(ctx)

        v_buffer = cl.Buffer(ctx,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=v.astype('f'))
        a_buffer = cl.Buffer(ctx,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a.astype('f'))
        points_buffer = cl.Buffer(ctx,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=points.astype('f'))
        weights_buffer = cl.Buffer(ctx,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=weights.astype('f'))
        #points_cache = cl.Buffer(ctx,cl.mem_flags.READ_WRITE, size=points.astype('f').nbytes)
        #weights_cache = cl.Buffer(ctx,cl.mem_flags.READ_WRITE, size=weights.astype('f').nbytes)
        output_buffer = cl.Buffer(ctx,cl.mem_flags.WRITE_ONLY, v.astype('f').nbytes)
        
        if kernel=="simplegaussquad":
            prg = cl.Program(ctx,r"""

#define OFFSET 0.5f /* (epsilon + 1.0-epsilon)/2.0 */
//#define DEGREE 300
//#define NUM_REPEATS 1000 /***!!! FOR PERFORMANCE TESTING ONLY... REDUCE THIS  DOWN TO 1 OTHERWISE! **/

float integrand(float x, float v, float a) 
{
  return pow(x,-1.5f)*pow(1.0f-x,-1.5f)*exp(-(1.0f+a*x)/(v*x*(1.0f-x)));
}

__kernel void integrate( __global const float *v,
                          __global const float *a,
                          __global const float *points_buf,
                          __global const float *weights_buf,
                          __local float *points,
                          __local float *weights,
                          __global float *output,
                          unsigned degree,
                          unsigned num_repeats)
{
  int gid = get_global_id(0);
  float epsilon;
  float my_v=v[gid], my_a=a[gid];
  float leadingcoeff_or_slope;
  float accum;
  int cnt;
  int lid = get_local_id(0);
  int lsize = get_local_size(0);
  int csize=degree/lsize;
  int repeatcnt;

  //(DEGREE/lsize)*(lsize)

  /* copy point and weight buffers into local cache */
  /* split load across local units */
  for (cnt=lid*csize; cnt < (lid+1)*csize;cnt++) {
    points[cnt]=points_buf[cnt];
  }
  for (cnt=lid*csize; cnt < (lid+1)*csize;cnt++) {
    weights[cnt]=weights_buf[cnt];
  }
  if (lid==lsize-1) { /* remainig few bytes */ 
    for (cnt=(lid+1)*csize; cnt < degree;cnt++) {
      points[cnt]=points_buf[cnt];
    }
    for (cnt=(lid+1)*csize; cnt < degree;cnt++) {
      weights[cnt]=weights_buf[cnt];
    }

  }
  barrier(CLK_LOCAL_MEM_FENCE);
  /* epsilon = .05/np.max((v,1.0,a)) */

  for (repeatcnt=0;repeatcnt < num_repeats;repeatcnt++) { /* repeating for performance testing */
  epsilon = 1.0f; 
  if (my_v > epsilon) {
    epsilon=my_v;
  } 
  if (my_a > epsilon) {
    epsilon=my_a;
  }
  epsilon = .05f/epsilon; 

  leadingcoeff_or_slope = (1.0f-2.0f*epsilon)/2.0f;

  /* vals = (1.0/v**2.0) * leadingcoefficient * np.tensordot(weights,np.vectorize(integrand)(slope*points+offset,v,a),axes=(0,0)) */

  accum=0.0f;     
  /* Note: On Intel GPU it gets quite a bit faster if degree is a constant rather than a parameter (!) */
  for (cnt=0;cnt < degree;cnt++) {
    accum += weights[cnt] * integrand(leadingcoeff_or_slope * points[cnt] + OFFSET,my_v,my_a);

  }
  accum *= leadingcoeff_or_slope/pow(my_v,2.0f);
  } /* End repeating */
  output[gid] = accum;
  
}        
        """).build()
            pass
        elif kernel=="quadpack":
            prg=cl.Program(ctx,r"""
// regenerate qagse_fparams.c with:
// f2c -a qagse_fparams.f
// patch -p0 <qagse_fparams.patch

#ifdef static 
#undef static
#endif

#define static const __constant // f2c generates static when it means const

// Force to single precision (qagpe is basically single precision anyway)
typedef float doublereal;
typedef float real;
typedef int integer;
typedef int logical;

#ifndef NULL
#define NULL ((char *)0)
#endif

int assert(int a) {
  char *null=NULL;
  if (!a) { 
    if (*null) return 0;// attempt to read from invalid address zero
  }
  return 1;
 }

//typedef real (*E_fp)();
typedef char *E_fp;  // Don't use this anymore... hardwired to funct(...)

float dabs(float p) { return fabs(p); }
float dmax(float p,float q) { if (p > q) return p;else return q; }
float dmin(float p,float q) { if (p < q) return p;else return q; }
//float min(float p,float q) { if (p < q) return p;else return q; }


doublereal pow_dd(doublereal *arg1,const __constant doublereal *arg2)
{
  return pow(*arg1,*arg2);
}

/* C source for R1MACH -- remove the * in column 1 */
float r1mach_(const __constant integer *i)
{
	switch(*i){
	  case 1: return FLT_MIN;
	  case 2: return FLT_MAX;
	  case 3: return FLT_EPSILON/FLT_RADIX;
	  case 4: return FLT_EPSILON;
	  case 5: return log10((float)FLT_RADIX);
	  }
//  fprintf(stderr, "invalid argument: r1mach(%ld)\n", *i);
	assert(0); return 0; /* else complaint of missing return value */
}
 


#define TRUE_ 1
#define FALSE_ 0


#include "qagse_fparams.c"


doublereal funct_(doublereal *x, doublereal *v, doublereal *a) 
{
  return pow(*x,-1.5f)*pow(1.0f-(*x),-1.5f)*exp(-(1.0f+(*a)*(*x))/((*v)*(*x)*(1.0f-(*x))));
}

#define LIMIT 50

__kernel void integrate( __global const float *v,
                          __global const float *a,
                          __global const float *points_buf, // does not use these
                          __global const float *weights_buf, // not used
                          __local float *points, // not used
                          __local float *weights, //not used
                          __global float *output,
                          unsigned degree,
                          unsigned num_repeats)
{
  int gid = get_global_id(0);
  float my_v=v[gid], my_a=a[gid];
  int cnt;
  int repeatcnt;
  float epsilon;
  float upper_bound;
  //float epsabs=1.49e-8,epsrel=1.49e-8; // defaults from scipy
  float epsabs=1.e-6,epsrel=1.e-4; // defaults from scipy
  float result=0.0,abserr=0.0;
  int neval=0;
  int ier=0;
  int limit=LIMIT;
  float alist[LIMIT];
  float blist[LIMIT];
  float rlist[LIMIT];
  float elist[LIMIT];
  int iord[LIMIT];
  int last;

  for (repeatcnt=0;repeatcnt < num_repeats;repeatcnt++) { /* repeating for performance testing */

    epsilon = 1.0f; 
    if (my_v > epsilon) {
      epsilon=my_v;
    } 
    if (my_a > epsilon) {
      epsilon=my_a;
    }
    epsilon = .05f/epsilon; 
    upper_bound=1.0f-epsilon;
    epsabs *= pow(my_v,2.0f);
    qagse_(NULL,&my_v,&my_a,&epsilon,&upper_bound,&epsabs,&epsrel,&limit,&result,&abserr,&neval,&ier,alist,blist,rlist,elist,iord,&last);

  } /* End repeating */
  output[gid] = result/pow(my_v,2.0f);
}        
""").build()
            pass
        else:
            raise ValueError("Unknown kernel %s" % (kernel))
        
        integrate_kernel=prg.integrate
        integrate_kernel.set_scalar_arg_dtypes([None,None,None,None,None,None,None, np.uint32,np.uint32])
        
        starttime=time.time()

        # WARNING: On beignet, if it takes more than about 6 seconds to
        # run it will SILENTLY FAIL
        # See http://stackoverflow.com/questions/27695807/opencl-timeout-on-beignet-doesnt-raise-error
        #
        # The 6 second limit can be removed with:
        #echo -n 0 > /sys/module/i915/parameters/enable_hangcheck
        # See: https://www.freedesktop.org/wiki/Software/Beignet/
        
        res=integrate_kernel(queue,(np.prod(v.shape),),None,v_buffer,a_buffer,points_buffer,weights_buffer,cl.LocalMemory(points.astype('f').nbytes),cl.LocalMemory(points.astype('f').nbytes),output_buffer,np.uint32(degree),np.uint32(num_repeats))
        
        gpuvals=np.empty(v.shape,dtype='f')
        cl.enqueue_copy(queue,gpuvals,output_buffer,wait_for=(res,),is_blocking=True)
        elapsed=time.time()-starttime
        
        gpuerr = np.abs(gpuvals-scipyvals[0])
        relgpuerr= gpuerr/np.abs(scipyvals[0])  # division by zero warning expected here
        
        print("GPU time for %d*%d iterations: %f" % (num_repeats,np.prod(v.shape),elapsed))
        
        GPUrate = num_repeats*np.prod(v.shape)/elapsed
        # For 24x22mm tiles, 0.5 mm point spacing
        # 24*22 -> 2112 surface points. Assume 1000 frames -> 2,112,000
        # matrix rows. by 1240 colums = 2.62 billion evals. 
        # Each

        # Laptop with Intel GPU gives compute time of ~9000 seconds
        # spec'd at ~ 325 Gflops
        
        computetime=(24.0*22.0/0.5/0.5)*1000.*1240./GPUrate
        print("Needed Compute time=%f s" % (computetime))

        print("Max GPUerror = %g" % (np.max(gpuerr)))
        print("Is relative error of %f%%" % (100.0*np.max(gpuerr)/np.abs(scipyvals[0].ravel()[np.argmax(gpuerr)])))

        pass
    
    
    
    if doplots:
        pl.figure(1)
        pl.clf()
        pl.imshow(np.log10(gpuvals+1e-6))
        pl.colorbar()

        pl.figure(2)
        pl.clf()
        pl.imshow(np.log10(vals+1e-6))
        pl.colorbar()

        pl.figure(3)
        pl.clf()
        pl.imshow(np.log10(scipyvals[0]+1e-6))
        pl.colorbar()
        
        pl.figure(4)
        pl.clf()
        pl.imshow(np.log10(oldvals[0]+1e-6))
        pl.colorbar()
    
        pl.figure(5)
        pl.clf()
        pl.imshow(np.log10(abs(err)))
        pl.colorbar()


        pl.figure(6)
        pl.clf()
        pl.imshow(np.log10(abs(relerr)))
        pl.colorbar()


        pl.figure(7)
        pl.clf()
        pl.imshow(np.log10(abs(gpuerr)))
        pl.colorbar()


        pl.figure(8)
        pl.clf()
        pl.imshow(np.log10(abs(relgpuerr)))
        pl.colorbar()


        pass
    return (gpuvals,vals,scipyvals[0],oldvals[0])


# for testing of newintegral
if __name__=="__main__":
    vrange=10**np.arange(-5,4,.1)
    crange=10**np.arange(0,4,.1)  # Note: Convergence trouble for small values of c... fortunately we don't care because c < 1 means 3d propagation is much shorter than 1D distance

    vrange=vrange.reshape(vrange.shape[0],1)
    crange=crange.reshape(1,crange.shape[0])

    v=vrange*np.ones((1,crange.shape[1]))
    c=crange*np.ones((vrange.shape[0],1))
    degree=300
    num_repeats=500
    # kernel="simplegaussquad"
    kernel="quadpack"
    (gpuvals,vals,scipyvals,oldvals)=newintegral(v,c,degree,num_repeats,kernel)
