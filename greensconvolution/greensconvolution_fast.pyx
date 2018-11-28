from __future__ import print_function

import sys
import os
import os.path

import numpy as np
from numpy.polynomial.legendre import leggauss

cimport numpy as np
from libc.math cimport log10
from libc.math cimport exp
from libc.math cimport sqrt
from libc.stdint cimport uint64_t
from libc.stdint cimport int64_t
from libc.stdint cimport int8_t

import greensconvolution.greensconvolution_calc

try:
    import pyopencl as cl
    pass
except:
    cl=None
    pass


cimport cython

pi=3.14159265358979323846

default_kernel="opencl_interpolator"
#default_kernel="openmp_interpolator"
#default_kernel="opencl_quadpack"
#default_kernel="opencl_simplegaussquad"


cdef extern from "greensconvolution_fast_c.h":
    void greensconvolution_integrate_anisotropic_c(
	float *vrange,uint64_t nvrange, #number of rows in integraleval
	float *crange,uint64_t ncrange, # number of cols in integraleval
	float *integraleval,
	float *integral_dintegranddveval, # same size as integraleval
	float *integral_dintegranddceval, # same size as integraleval
	float *zvec, uint64_t *zvecshape,uint64_t *zvecstrides,
	float *xvec, uint64_t *xvecshape,uint64_t *xvecstrides,
	float *tvec, uint64_t *tvecshape,uint64_t *tvecstrides,
        float yval,
        float *curvaturevec, uint64_t *curvaturevecshape, uint64_t *curvaturevecstrides,
        uint64_t *sumstrides, uint64_t *shape,
        double alphaz, double alphaxy, int8_t curvature_flag,
	float *result,uint64_t *resultstrides,
	double coeff,
	uint64_t *axissumflag,
	uint64_t ndim)  
    pass


def greensconvolution_crosscheck(greensconvolution_params,z,x,t,y,k,rho,cp):

    # first calculate with greensconvolution_calc.evalfromneighbors()
    coeff=2.0/((rho*cp*pi*pi)*z**3.0)
    c=np.abs(np.sqrt(z**2 + y**2 + x**2)/z)
    alpha=k*1.0/(rho*cp)
    v=(4*alpha*t)/z**2.0

    vrange=greensconvolution_params.vrange
    crange=greensconvolution_params.crange
    integraleval=greensconvolution_params.integraleval
    integral_dintegranddveval=greensconvolution_params.integral_dintegranddveval
    integral_dintegranddceval=greensconvolution_params.integral_dintegranddceval
    
    res=coeff*greensconvolution.greensconvolution_calc.evalfromneighbors(vrange,crange,integraleval,integral_dintegranddveval,integral_dintegranddceval,v,c,verbose=True)

    # now evaluate with greensconvolution_integrate
    z=np.array((z,),dtype='f')
    x=np.array((x,),dtype='f')
    t=np.array((t,),dtype='f')

    res2=greensconvolution_integrate(greensconvolution_params,z,x,t,y,k,rho,cp,1.0,())

    print("greensconvolution_crosscheck: %f vs. %f" % (res,res2))

    return(res,res2)


def greensconvolution_integrate(greensconvolution_params,zvec,xvec,tvec,y,k,rho,cp,coeff,sumaxes,kernel=default_kernel):
    # zvec, xvec, and tvec should have the same number of dimensions, and
    # this function broadcasts over singleton dimensions and then
    # sums over the axes specified in sumaxes.
    
    # zvec is depths from surface (initial propagation)
    # xvec is horizontal distance of measurement from scatterer (scattered propagation)
    # tvec may have a different length and the result will be an array
    # print(zvec.shape) 
    return greensconvolution_integrate_anisotropic(greensconvolution_params,zvec,xvec,tvec,y,k,k,k,rho,cp,coeff,sumaxes,kernel=kernel)


def greensconvolution_integrate_py(greensconvolution_params,np.ndarray[double,ndim=1] zvec,np.ndarray[double,ndim=1] xvec,np.ndarray[double,ndim=1] tvec,yval,k,rho,cp):
    # WARNING: Still has old API
    # zvec and rvec should have the same length, and the result will
    # be integrated over them
    # zvec is depths from surface (initial propagation)
    # xvec is horizontal distance of measurement from scatterer (scattered propagation)
    # tvec may have a different length and the result will be an array
    return greensconvolution_integrate_anisotropic_py(greensconvolution_params,zvec,xvec,tvec,yval,k,k,k,rho,cp)



#@cython.boundscheck(False)
def greensconvolution_integrate_anisotropic_py(greensconvolution_params,np.ndarray[double,ndim=1] zvec,np.ndarray[double,ndim=1] xvec,np.ndarray[double,ndim=1] tvec,yval,kz,ky,kx,rho,cp):
    # WARNING: Still has old API
    
    # zvec and xvec should have the same length, and the result will
    # be integrated over them
    # zvec is depths from surface (initial propagation)
    # x is horizontal distance of measurement from scatterer (scattered propagation).
    # (OLD) rconductivityscaledvec should be sqrt(x^2*(kz/kx) + y^2*(kz/ky)+z^2)
    # tvec may have a different length and the result will be an array

    # NOTE: This is supposed to be the same code as in greensconvolution_fast_c.c/greensconvolution_integrate_anisotropic_c.c
    cdef np.ndarray[double,ndim=2] vrange
    cdef np.ndarray[double,ndim=2] crange
    cdef np.ndarray[double,ndim=2] integraleval
    cdef np.ndarray[double,ndim=2] integral_dintegranddveval
    cdef np.ndarray[double,ndim=2] integral_dintegranddceval

    cdef np.ndarray[double,ndim=1] rconductivityscaledvec = np.sqrt((xvec**2.0)*(kz/kx) + (yval**2.0)*(kz/ky) + (zvec**2.0))
    
    cdef np.ndarray[double,ndim=1] coeff
    cdef np.ndarray[double,ndim=1] v
    cdef np.ndarray[double,ndim=1] c
    cdef uint64_t count
    cdef double log10v0,log10c0,dlog10v,dlog10c,vval,cval
    cdef int64_t vidx,vidx2,cidx,cidx2
    cdef int64_t point_vidx[4]
    cdef int64_t point_cidx[4]
    cdef double vidxval,cidxval,integralevalpt,integral_dintegranddvevalpt,integral_dintegranddcevalpt
    cdef double vals[4],
    cdef double weights[4]
    cdef double totalweight,est
    cdef int pointcnt
    cdef double accum
    cdef np.ndarray[double,ndim=1] result
    cdef int64_t tcnt,nt
    cdef double alpha
    
    vrange=greensconvolution_params.vrange
    crange=greensconvolution_params.crange
    integraleval=greensconvolution_params.integraleval
    integral_dintegranddveval=greensconvolution_params.integral_dintegranddveval
    integral_dintegranddceval=greensconvolution_params.integral_dintegranddceval
    
    #alphax=kx*1.0/(rho*cp)
    #alphay=ky*1.0/(rho*cp)
    alphaz=kz*1.0/(rho*cp)

    alphaxyz=((kx*ky*kz)**(1.0/3.0))/(rho*cp)
    
    coeff=2.0*(alphaz**(3.0/2.0))/((rho*cp*pi*pi)*(alphaxyz**(3.0/2.0))*(zvec**3.0))

    nvrange=vrange.shape[0]
    ncrange=crange.shape[1]
    assert(vrange.shape[1]==1)
    assert(crange.shape[0]==1)

    
    log10v0=log10(vrange[0,0])
    log10c0=log10(crange[0,0])
    dlog10v=log10(vrange[1,0])-log10(vrange[0,0])
    dlog10c=log10(crange[0,1])-log10(crange[0,0])

    nz=zvec.shape[0]
    assert(nz==rconductivityscaledvec.shape[0])
    
    nt=tvec.shape[0]
    result=np.zeros(nt,dtype='d')
    
    c=np.abs(rconductivityscaledvec/zvec)
    for tcnt in range(nt):

        assert(tvec[tcnt] > 0)
        
        v=(4*alphaz*tvec[tcnt])/zvec**2.0
        
        
    
        accum=0.0

        for count in range(nz):
            vval=v[count]            
            cval=c[count]

            # print("%f, %f, %f" % (log10(vval),log10v0,dlog10v))
            
            vidx=int((log10(vval)-log10v0)/dlog10v)
            # print("vidx=%d; nvrange=%d" % (vidx,nvrange))
            assert(vidx >= 0 and vidx+1 < nvrange) 
            vidx2=vidx+1
            
            
            cidx=int((log10(cval)-log10c0)/dlog10c)
            # print("cidx=%d; ncrange=%d" % (cidx,ncrange))
            assert(cidx >= 0 and cidx+1 < ncrange) 
            cidx2=cidx+1
            
            point_vidx[0]=vidx
            point_cidx[0]=cidx
            
            point_vidx[1]=vidx
            point_cidx[1]=cidx2

            point_vidx[2]=vidx2
            point_cidx[2]=cidx

            point_vidx[3]=vidx2
            point_cidx[3]=cidx2

            totalweight=0.0
        
            for pointcnt in range(4):
                vidxval=vrange[point_vidx[pointcnt],0]
                cidxval=crange[0,point_cidx[pointcnt]]
            
                integralevalpt=integraleval[point_vidx[pointcnt],point_cidx[pointcnt]]
                integral_dintegranddvevalpt=integral_dintegranddveval[point_vidx[pointcnt],point_cidx[pointcnt]]
                integral_dintegranddcevalpt=integral_dintegranddceval[point_vidx[pointcnt],point_cidx[pointcnt]]
                vals[pointcnt] = integralevalpt + (vval-vidxval)*integral_dintegranddvevalpt +(cval-cidxval)*integral_dintegranddcevalpt;
                weights[pointcnt]=sqrt(1.0/(0.001+(vval-vidxval)*(vval-vidxval) + (cval-cidxval)*(cval-cidxval)))
                totalweight+=weights[pointcnt]
            
                pass
            
            est=0.0
            for pointcnt in range(4):
                est+=vals[pointcnt]*weights[pointcnt]/totalweight
                pass

            # Limit according to nonnegative and upper bound in greensfcn_doc.tex
            if est < 0.0:
                # print("Warning: Integral gave inaccurate calculation of %g at v=%g, c=%g; lower bound of 0 used instead" % (est,vval,cval),file=sys.stderr)
                est=0.0
                pass
            elif est > 0.185*exp(-(cval**2.0-1.0)/vval):
                print("Warning: Integral gave inaccurate calculation of %g at v=%g,c=%g; upper bound of %g used instead" % (est,vval,cval,0.185*exp(-(cval**2.0-1.0)/vval)),file=sys.stderr)
                est= 0.185*exp(-(cval**2.0-1.0)/vval)
                pass
            
            accum+=coeff[count]*est
            pass
        result[tcnt]=accum
        pass
    return result



def LoadGCKernel(greensconvolution_params,kernel):
    cdef np.ndarray[float,ndim=2,mode="c"] vrange
    cdef np.ndarray[float,ndim=2,mode="c"] crange

    cdef np.ndarray[float,ndim=2,mode="c"] integraleval
    cdef np.ndarray[float,ndim=2,mode="c"] integral_dintegranddveval
    cdef np.ndarray[float,ndim=2,mode="c"] integral_dintegranddceval

    cdef np.ndarray[float,ndim=1,mode="c"] points_float
    cdef np.ndarray[float,ndim=1,mode="c"] weights_float

    if (kernel=="opencl_interpolator" or kernel=="opencl_interpolator_curved") and greensconvolution_params.opencl_interpolator is None:
        
        filename1=os.path.join(os.path.dirname(__file__),"opencl_interpolator_prefix.c")
        fh1=open(filename1,"r")
        
        filename2=os.path.join(os.path.dirname(__file__),"greensconvolution_fast_c.c")
        fh2=open(filename2,"r")

        #filename3=os.path.join(os.path.dirname(__file__),"opencl_interpolator_funcall.c")
        #fh3=open(filename3,"r")


        kernel_source=fh1.read()+fh2.read() #+fh3.read()
        fh1.close()
        fh2.close()
        #fh3.close()

        greensconvolution_params.opencl_interpolator = cl.Program(greensconvolution_params.OpenCL_CTX,kernel_source).build()

        greensconvolution_params.opencl_interpolator_function=greensconvolution_params.opencl_interpolator.greensconvolution_integrate_anisotropic_c_opencl
        greensconvolution_params.opencl_interpolator_function.set_scalar_arg_dtypes([None,np.uint64,
                                                                                    None,np.uint64,
                                                                                    None,
                                                                                    None,
                                                                                    None,
                                                                                    None, None, None,
                                                                                    None, None, None,
                                                                                    None, None, None,
                                                                                     np.float32,
                                                                                     None, None,
                                                                                     np.float32,np.float32,
                                                                                    None,None,
                                                                                    np.float32,
                                                                                    None,
                                                                                    np.uint64])


        greensconvolution_params.opencl_interpolator_curved_function=greensconvolution_params.opencl_interpolator.greensconvolution_integrate_anisotropic_curved_c_opencl
        greensconvolution_params.opencl_interpolator_curved_function.set_scalar_arg_dtypes([None,np.uint64,
                                                                                            None,np.uint64,
                                                                                            None,
                                                                                            None,
                                                                                            None,
                                                                                            None, None, None,
                                                                                            None, None, None,
                                                                                            None, None, None,
                                                                                            np.float32,
                                                                                            None, None, None,
                                                                                            None, None,
                                                                                            np.float32,np.float32,
                                                                                            None,None,
                                                                                            np.float32,
                                                                                            None,
                                                                                            np.uint64])

        
        vrange=greensconvolution_params.vrange
        greensconvolution_params.opencl_interpolator_vrange_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=vrange)

        crange=greensconvolution_params.crange
        greensconvolution_params.opencl_interpolator_crange_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=crange)
        
        integraleval=greensconvolution_params.integraleval
        greensconvolution_params.opencl_interpolator_integraleval_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=integraleval)
        
        integral_dintegranddveval=greensconvolution_params.integral_dintegranddveval
        greensconvolution_params.opencl_interpolator_integral_dintegranddveval_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=integral_dintegranddveval)
        
        integral_dintegranddceval=greensconvolution_params.integral_dintegranddceval
        greensconvolution_params.opencl_interpolator_integral_dintegranddceval_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=integral_dintegranddceval)

        

        
        pass


    if kernel=="opencl_simplegaussquad" and greensconvolution_params.opencl_simplegaussquad is None:
        
        filename1=os.path.join(os.path.dirname(__file__),"simplegaussquad.c")
        fh1=open(filename1,"r")
        
        degree=300

        kernel_source=("#define DEGREE %d\n" % (degree)) + fh1.read()
        fh1.close()

        greensconvolution_params.opencl_simplegaussquad = cl.Program(greensconvolution_params.OpenCL_CTX,kernel_source).build()

        greensconvolution_params.opencl_simplegaussquad_function=greensconvolution_params.opencl_simplegaussquad.simplegaussquad_opencl
        greensconvolution_params.opencl_simplegaussquad_function.set_scalar_arg_dtypes([None,
                                                                               None,
                                                                               None, None, None,
                                                                               None, None, None,
                                                                               None, None, None,
                                                                                        np.float32,
                                                                               None, None,
                                                                               np.float32,
                                                                               None,None,
                                                                               np.float32,
                                                                               None,
                                                                               np.uint64])
        
        (points,weights)=leggauss(degree)
        points_float=points.astype(np.float32) 
        weights_float=weights.astype(np.float32) 
        
        # Worst case behavior around a=00, v=50.0... corresponds to very early time, directly over reflector
        greensconvolution_params.opencl_simplegaussquad_points=points_float
        greensconvolution_params.opencl_simplegaussquad_points_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=greensconvolution_params.opencl_simplegaussquad_points)


        greensconvolution_params.opencl_simplegaussquad_weights=weights_float
        greensconvolution_params.opencl_simplegaussquad_weights_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=greensconvolution_params.opencl_simplegaussquad_weights)
        

        
        pass



    if kernel=="opencl_quadpack" and greensconvolution_params.opencl_quadpack is None:
        
        filename1=os.path.join(os.path.dirname(__file__),"quadpack_prefix.c")
        fh1=open(filename1,"r")
        
        filename2=os.path.join(os.path.dirname(__file__),"qagse_fparams.c")
        fh2=open(filename2,"r")

        filename3=os.path.join(os.path.dirname(__file__),"quadpack.c")
        fh3=open(filename3,"r")


        kernel_source=fh1.read()+fh2.read()+fh3.read()
        fh1.close()
        fh2.close()
        fh3.close()
    
        greensconvolution_params.opencl_quadpack = cl.Program(greensconvolution_params.OpenCL_CTX,kernel_source).build()
        
        greensconvolution_params.opencl_quadpack_function=greensconvolution_params.opencl_quadpack.quadpack_opencl
        greensconvolution_params.opencl_quadpack_function.set_scalar_arg_dtypes([              
            None, None, None,
            None, None, None,
            None, None, None,
            np.float32,
            None, None,
            np.float32,
            np.float32,
            None,None,
            np.float32,
            None,
            np.uint64])
                

        

    if kernel=="opencl_imagesources" and greensconvolution_params.opencl_imagesources is None:
        filename1=os.path.join(os.path.dirname(__file__),"imagesources.c")
        fh1=open(filename1,"r")
        
        kernel_source=fh1.read()
        fh1.close()

        greensconvolution_params.opencl_imagesources=cl.Program(greensconvolution_params.OpenCL_CTX,kernel_source).build()
        greensconvolution_params.opencl_imagesources_function=greensconvolution_params.opencl_imagesources.imagesources_opencl
        greensconvolution_params.opencl_imagesources_function.set_scalar_arg_dtypes([              
            None, None, None,
            None, None, None,
            None, 
            np.float32,
            None, None,
            np.float32,
            np.uint64,
            None,
            np.uint64])
        

        
        pass



    if kernel=="opencl_imagesources_curved" and greensconvolution_params.opencl_imagesources_curved is None:
        filename1=os.path.join(os.path.dirname(__file__),"imagesources_curved.c")
        fh1=open(filename1,"r")
        
        kernel_source=fh1.read()
        fh1.close()
        
        greensconvolution_params.opencl_imagesources_curved=cl.Program(greensconvolution_params.OpenCL_CTX,kernel_source).build()
        greensconvolution_params.opencl_imagesources_curved_function=greensconvolution_params.opencl_imagesources_curved.imagesources_curved_opencl
        greensconvolution_params.opencl_imagesources_curved_function.set_scalar_arg_dtypes([              
            None, None, None,
            None, None, None,
            None, None, None,
            None, 
            np.float32,
            None, None,
            np.float32,
            np.uint64,
            None,
            np.uint64])
        
        
        
        pass

    if kernel=="opencl_greensfcn_curved" and greensconvolution_params.opencl_greensfcn_curved is None:
        filename1=os.path.join(os.path.dirname(__file__),"greensfcn_curved.c")
        fh1=open(filename1,"r")
        
        kernel_source=fh1.read()
        fh1.close()
        
        greensconvolution_params.opencl_greensfcn_curved=cl.Program(greensconvolution_params.OpenCL_CTX,kernel_source).build()
        greensconvolution_params.opencl_greensfcn_curved_function=greensconvolution_params.opencl_greensfcn_curved.greensfcn_curved_opencl
        greensconvolution_params.opencl_greensfcn_curved_function.set_scalar_arg_dtypes([              
            None, None, None,  # linelength...
            None, None, None,  # tvec...
            None, None, None,  # source_intensity...
            None, None, None,  # depth...
            None, None, None,  # theta...
            None, None, None,  # avgcurvatures...
            None, None, None,  # avgcrosscurvatures...
	    None, None, None,  # iop_dy... (may be NULL)
	    None, None, None,  # iop_dx... (may be NULL)
            None, 
            None,
            None,
            np.float32, np.float32, np.float32,np.float32,np.float32,
            None, None, None,
            np.uint64])
        
        
        
        pass


    
    
    
    pass
        

def greensconvolution_image_sources(greensconvolution_params,rconductivityscaledvec_no_z_input,tvecinput,image_source_zposns_input,kz,rho,cp,coeff,sumaxes,avgcurvatures=None,kxy=None,opencl_queue=None):
    # image source order should be over last axis of rconductivityscaledvecinput, tvecinput, which should match imageorders in length
    # implicit sum over imageorders, which should be the last axis
    
    cdef np.ndarray[float,mode="c"] rconductivityscaledvec_no_z=rconductivityscaledvec_no_z_input.reshape(np.prod(rconductivityscaledvec_no_z_input.shape))
    cdef np.ndarray[float,mode="c"] tvec=tvecinput.reshape(np.prod(tvecinput.shape))
    cdef np.ndarray[float,mode="c"] avgcurvatures_cython
    
    cdef np.ndarray[int64_t,ndim=1,mode="c"] tvecshape 
    cdef np.ndarray[int64_t,ndim=1,mode="c"] rvecshape
    cdef np.ndarray[int64_t,ndim=1,mode="c"] avgcurvaturesshape
    cdef np.ndarray[float,ndim=1,mode="c"] image_source_zposns=image_source_zposns_input
    
    cdef np.ndarray[int64_t,ndim=1,mode="c"] tvecstrides 
    cdef np.ndarray[int64_t,ndim=1,mode="c"] rvecstrides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] avgcurvaturesstrides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] resultstrides

    
    
    cdef np.ndarray[float,mode="c"] result

    cdef np.ndarray[int64_t,ndim=1,mode="c"] shape
    cdef np.ndarray[float,ndim=1,mode="c"] float_zero=np.array((0.0,),dtype='f')


    cdef float alphaz=kz*1.0/(rho*cp)
    cdef float alphaxy=alphaz
    cdef float alphaxyz
    
    if kxy is not None:
        alphaxy = kxy*1.0/(rho*cp)
        pass

    alphaxyz = ((alphaxy**2.0)*alphaz)**(1.0/3.0)
    coeff_prod=coeff*(2.0/(rho*cp))/((4.0*np.pi*(alphaxyz))**(3.0/2.0))
    
    tvecshape=np.array(tvecinput.shape,dtype=np.int64)
    tvecstrides=np.array(np.cumprod(np.concatenate((tvecshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)
    rvecshape=np.array(rconductivityscaledvec_no_z_input.shape,dtype=np.int64)
    rvecstrides=np.array(np.cumprod(np.concatenate((rvecshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)


    if avgcurvatures is not None:
        avgcurvatures_cython=avgcurvatures.reshape(np.prod(avgcurvatures.shape))
        avgcurvaturesshape=np.array(avgcurvatures.shape,dtype=np.int64)
        avgcurvaturesstrides=np.array(np.cumprod(np.concatenate((avgcurvaturesshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)
        pass
    

    shape=np.array((tvecshape,rvecshape),dtype=np.int64).max(0)
    ndim=len(shape)

    resultshape=np.array(shape,dtype=np.int64)  
    resultstrides = np.array(np.cumprod(np.concatenate((resultshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)

    PySumAxes=sumaxes
    PyAxisSumFlag=np.zeros(ndim,dtype=np.int64)
    if len(PySumAxes) > 0:
        #print(sumaxes)
        #print(axissumflag)
        PyAxisSumFlag[np.array(PySumAxes)]=1  # Numpy treats an array as a single index, whereas a tuple would be a sequence of indices, which isn't what we want here.
        pass

    iterlen=np.prod(resultshape)

    if greensconvolution_params.OpenCL_Version < 1.2:
        # Cannot use fill... copy zero buffer instead
        result=np.zeros(np.prod(resultshape),dtype='f')
        pass
    else:
        result=np.empty(np.prod(resultshape),dtype='f')
        pass

    if avgcurvatures is None:
        kernel="opencl_imagesources"
        pass
    else:
        kernel="opencl_imagesources_curved"
        pass

    
    LoadGCKernel(greensconvolution_params, kernel)
    
    queue=opencl_queue
    if queue is None:
        queue=cl.CommandQueue(greensconvolution_params.OpenCL_CTX,properties=greensconvolution.greensconvolution_calc.OpenCL_GetOutOfOrderDeviceQueueProperties(greensconvolution_params.OpenCL_CTX))
        pass
    assert(queue is not None)
        
    rvec_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=rconductivityscaledvec_no_z)
    rvecshape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=rvecshape)
    rvecstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=rvecstrides)
    
    tvec_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=tvec)
    tvecshape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=tvecshape)
    tvecstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=tvecstrides)

    if avgcurvatures is not None:
        avgcurvatures_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=avgcurvatures_cython)
        avgcurvaturesshape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=avgcurvaturesshape)
        avgcurvaturesstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=avgcurvaturesstrides)

        pass
    
    

    shape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=shape)
    result_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_WRITE,size=result.nbytes)
    #print("allocated result buffer.")
    resultstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=resultstrides)

    image_source_zposns_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=image_source_zposns)

    if greensconvolution_params.OpenCL_Version < 1.2:
        fill_event=cl.enqueue_copy(queue,result_buffer,result,is_blocking=False);
        pass
    else:
        fill_event=cl.enqueue_fill_buffer(queue,result_buffer,float_zero,0,result.nbytes);
        pass


    if avgcurvatures is None:
        # opencl_imagesources kernel     
        kernel_event=greensconvolution_params.opencl_imagesources_function(queue,(iterlen,),None,
                                                                           rvec_buffer,rvecshape_buffer,rvecstrides_buffer,
                                                                           tvec_buffer,tvecshape_buffer,tvecstrides_buffer,
                                                                           
                                                                           shape_buffer,
                                                                           alphaz,
                                                                           result_buffer,resultstrides_buffer,
                                                                           coeff_prod,
                                                                           ndim,
                                                                           image_source_zposns_buffer,
                                                                           image_source_zposns.shape[0],
                                                                           wait_for=(fill_event,))
        pass
    else:
        # opencl_imagesources_curved kernel
	#alphaxy=kxy*1.0/(rho*cp)
        #print("coeff_prod=%f" % (coeff_prod))
        kernel_event=greensconvolution_params.opencl_imagesources_curved_function(queue,(iterlen,),None,
                                                                                  rvec_buffer,rvecshape_buffer,rvecstrides_buffer,
                                                                                  tvec_buffer,tvecshape_buffer,tvecstrides_buffer,
                                                                                  
                                                                                  avgcurvatures_buffer,avgcurvaturesshape_buffer,avgcurvaturesstrides_buffer,
                                                                                  shape_buffer,
                                                                                  alphaz,
                                                                                  result_buffer,resultstrides_buffer,
                                                                                  coeff_prod,
                                                                                  ndim,
                                                                                  image_source_zposns_buffer,
                                                                                  image_source_zposns.shape[0],
                                                                                  wait_for=(fill_event,))
        
        pass
    copyout_event=cl.enqueue_copy(queue,result,result_buffer,wait_for=(kernel_event,),is_blocking=False)

    
    resultref=returnparam(result)  # python (not cython) reference that we can access from closure
    returnshape=returnparam(shape) #returnparam(shape)
    PySummedShape=shape[(~PyAxisSumFlag.astype(np.bool))]

    # Closure function is what we have to do to clean up. and get data
    def closure():
        copyout_event.wait()
        
        
        rvec_buffer.release()
        rvecshape_buffer.release()
        rvecstrides_buffer.release()
        
        tvec_buffer.release()
        tvecshape_buffer.release()
        tvecstrides_buffer.release()
        
        if avgcurvatures is not None:
            avgcurvatures_buffer.release()
            avgcurvaturesshape_buffer.release()
            avgcurvaturesstrides_buffer.release()
            pass
        
        result_buffer.release()
        resultstrides_buffer.release()
        shape_buffer.release()
        
        
        
        if len(PySumAxes) > 0:
            #sys.stderr.write("resultref.shape=%s\n" % (str(resultref.shape)))
            #sys.stderr.write("shaperef=%s\n" % (str(shaperef)))
            #sys.stderr.write("PySumAxes=%s\n" % (str(PySumAxes)))
            #sys.stderr.write("PySummedShape=%s\n" % (str(PySummedShape)))
            #sys.stderr.flush()
            return resultref.reshape(*returnshape).sum(PySumAxes).reshape(*PySummedShape)
        else:
            return resultref.reshape(*PySummedShape)
        pass
            
    if opencl_queue is not None:
        # queue parameter specified... return closure

        return closure
    queue.finish()
    # Otherwise, execute closure, freeing memory immediately and performing summation
    return closure()
    pass



def greensconvolution_greensfcn_curved(greensconvolution_params,source_intensity_input,linelength_input,depth_input,theta_input,tvecinput,kz,rho,cp,sumaxes,avgcurvatures=None,avgcrosscurvatures=None,iop_dy=None,iop_dx=None,ky=None,kx=None,opencl_queue=None):
    # image source order should be over last axis of rconductivityscaledvecinput, tvecinput, which should match imageorders in length
    # implicit sum over imageorders, which should be the last axis
    # iop_dx and iop_dy are step sizes for integrating over the pixel
    
    cdef np.ndarray[float,mode="c"] linelength=linelength_input.reshape(np.prod(linelength_input.shape))
    cdef np.ndarray[float,mode="c"] depth=depth_input.reshape(np.prod(depth_input.shape))
    cdef np.ndarray[float,mode="c"] theta=theta_input.reshape(np.prod(theta_input.shape))
    cdef np.ndarray[float,mode="c"] source_intensity=source_intensity_input.reshape(np.prod(source_intensity_input.shape))
    cdef np.ndarray[float,mode="c"] tvec=tvecinput.reshape(np.prod(tvecinput.shape))
    cdef np.ndarray[float,mode="c"] avgcurvatures_cython
    cdef np.ndarray[float,mode="c"] avgcrosscurvatures_cython
    cdef np.ndarray[float,mode="c"] iop_dy_cython
    cdef np.ndarray[float,mode="c"] iop_dx_cython
    
    cdef np.ndarray[int64_t,ndim=1,mode="c"] tvecshape 
    cdef np.ndarray[int64_t,ndim=1,mode="c"] linelengthshape
    cdef np.ndarray[int64_t,ndim=1,mode="c"] depthshape
    cdef np.ndarray[int64_t,ndim=1,mode="c"] thetashape
    cdef np.ndarray[int64_t,ndim=1,mode="c"] source_intensityshape
    cdef np.ndarray[int64_t,ndim=1,mode="c"] avgcurvaturesshape
    cdef np.ndarray[int64_t,ndim=1,mode="c"] avgcrosscurvaturesshape
    cdef np.ndarray[int64_t,ndim=1,mode="c"] iop_dy_shape
    cdef np.ndarray[int64_t,ndim=1,mode="c"] iop_dx_shape
    
    cdef np.ndarray[int64_t,ndim=1,mode="c"] tvecstrides 
    cdef np.ndarray[int64_t,ndim=1,mode="c"] linelengthstrides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] depthstrides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] thetastrides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] source_intensitystrides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] avgcurvaturesstrides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] avgcrosscurvaturesstrides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] iop_dy_strides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] iop_dx_strides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] resultstrides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] axissumflag 

    cdef np.ndarray[int64_t,ndim=1,mode="c"] sumstrides
    
    
    cdef np.ndarray[float,mode="c"] result

    cdef np.ndarray[int64_t,ndim=1,mode="c"] shape
    cdef np.ndarray[float,ndim=1,mode="c"] float_zero=np.array((0.0,),dtype='f')


    cdef float alphaz=kz*1.0/(rho*cp)
    cdef float alphaxy
    
    tvecshape=np.array(tvecinput.shape,dtype=np.int64)
    tvecstrides=np.array(np.cumprod(np.concatenate((tvecshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)
    linelengthshape=np.array(linelength_input.shape,dtype=np.int64)
    linelengthstrides=np.array(np.cumprod(np.concatenate((linelengthshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)
    depthshape=np.array(depth_input.shape,dtype=np.int64)
    depthstrides=np.array(np.cumprod(np.concatenate((depthshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)
    thetashape=np.array(theta_input.shape,dtype=np.int64)
    thetastrides=np.array(np.cumprod(np.concatenate((thetashape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)

    source_intensityshape=np.array(source_intensity_input.shape,dtype=np.int64)
    source_intensitystrides=np.array(np.cumprod(np.concatenate((source_intensityshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)


    if ky is None:
        ky=kz
        pass
    
    if kx is None:
        kx=kz
        pass

    if avgcurvatures is not None:
        assert(avgcrosscurvatures is not None)
        avgcurvatures_cython=avgcurvatures.reshape(np.prod(avgcurvatures.shape))
        avgcurvaturesshape=np.array(avgcurvatures.shape,dtype=np.int64)
        avgcurvaturesstrides=np.array(np.cumprod(np.concatenate((avgcurvaturesshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)

        avgcrosscurvatures_cython=avgcrosscurvatures.reshape(np.prod(avgcrosscurvatures.shape))
        avgcrosscurvaturesshape=np.array(avgcrosscurvatures.shape,dtype=np.int64)
        avgcrosscurvaturesstrides=np.array(np.cumprod(np.concatenate((avgcrosscurvaturesshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)

        pass
    
    if iop_dx is not None or iop_dy is not None:
        assert(iop_dx is not None and iop_dy is not None)
    
        iop_dy_cython=iop_dy.reshape(np.prod(iop_dy.shape))
        iop_dy_shape=np.array(iop_dy.shape,dtype=np.int64)
        iop_dy_strides=np.array(np.cumprod(np.concatenate((iop_dy_shape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)

        iop_dx_cython=iop_dx.reshape(np.prod(iop_dx.shape))
        iop_dx_shape=np.array(iop_dx.shape,dtype=np.int64)
        iop_dx_strides=np.array(np.cumprod(np.concatenate((iop_dx_shape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)
        pass


    shape=np.array((tvecshape,linelengthshape,source_intensityshape),dtype=np.int64).max(0)
    ndim=len(shape)

    axissumflag=np.zeros(ndim,dtype=np.int64)
    if len(sumaxes) > 0:
        axissumflag[np.array(sumaxes)]=1
        pass
    
    
    resultshape=np.array(shape,dtype=np.int64)  
    resultshape[axissumflag.astype(np.bool)]=1
    resultstrides = np.array(np.cumprod(np.concatenate((resultshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)

    resultsummedshape=resultshape[(~axissumflag.astype(np.bool))]

    sumshape=np.array(shape,dtype=np.int64)  
    sumshape[~axissumflag.astype(np.bool)]=1
    sumstrides=np.array(np.cumprod(np.concatenate((sumshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)




    iterlen=np.prod(resultshape)

    if greensconvolution_params.OpenCL_Version < 1.2:
        # Cannot use fill... copy zero buffer instead
        result=np.zeros(np.prod(resultshape),dtype='f')
        pass
    else:
        result=np.empty(np.prod(resultshape),dtype='f')
        pass

    if avgcurvatures is None:
        kernel="opencl_greensfcn"
        pass
    else:
        kernel="opencl_greensfcn_curved"
        pass

    
    LoadGCKernel(greensconvolution_params, kernel)
    
    queue=opencl_queue
    if queue is None:
        queue=cl.CommandQueue(greensconvolution_params.OpenCL_CTX,properties=greensconvolution.greensconvolution_calc.OpenCL_GetOutOfOrderDeviceQueueProperties(greensconvolution_params.OpenCL_CTX))
        pass
    assert(queue is not None)
        
    linelength_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=linelength)
    linelengthshape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=linelengthshape)
    linelengthstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=linelengthstrides)

    depth_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=depth)
    depthshape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=depthshape)
    depthstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=depthstrides)

    theta_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=theta)
    thetashape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=thetashape)
    thetastrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=thetastrides)

    
    tvec_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=tvec)
    tvecshape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=tvecshape)
    tvecstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=tvecstrides)

    source_intensity_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=source_intensity)
    source_intensityshape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=source_intensityshape)
    source_intensitystrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=source_intensitystrides)

    if avgcurvatures is not None:
        avgcurvatures_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=avgcurvatures_cython)
        avgcurvaturesshape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=avgcurvaturesshape)
        avgcurvaturesstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=avgcurvaturesstrides)

        avgcrosscurvatures_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=avgcrosscurvatures_cython)
        avgcrosscurvaturesshape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=avgcrosscurvaturesshape)
        avgcrosscurvaturesstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=avgcrosscurvaturesstrides)

        pass
    
    if iop_dy is not None:
        iop_dy_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=iop_dy_cython)
        iop_dy_shape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=iop_dy_shape)
        iop_dy_strides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=iop_dy_strides)

        iop_dx_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=iop_dx_cython)
        iop_dx_shape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=iop_dx_shape)
        iop_dx_strides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=iop_dx_strides)
        pass
    else:
        iop_dy_buffer=None
        iop_dy_shape_buffer=None
        iop_dy_strides_buffer=None

        iop_dx_buffer=None
        iop_dx_shape_buffer=None
        iop_dx_strides_buffer=None
        pass

    shape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=shape)
    axissumflag_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=axissumflag)
    resultshape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=resultshape)

    sumstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=sumstrides)


    result_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_WRITE,size=result.nbytes)
    #print("allocated result buffer.")
    resultstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=resultstrides)

    if greensconvolution_params.OpenCL_Version < 1.2:
        fill_event=cl.enqueue_copy(queue,result_buffer,result,is_blocking=False);
        pass
    else:
        fill_event=cl.enqueue_fill_buffer(queue,result_buffer,float_zero,0,result.nbytes);
        pass


    if avgcurvatures is None:
        # opencl_greensfcn kernel     
        # *** NOT YET IMPLEMENTED ***
        kernel_event=greensconvolution_params.opencl_greensfcn_function(queue,(iterlen,),None,
                                                                        linelength_buffer,linelengthshape_buffer,linelengthstrides_buffer,
                                                                        tvec_buffer,tvecshape_buffer,tvecstrides_buffer,
                                                                           
                                                                        source_intensity_buffer,source_intensityshape_buffer,source_intensitystrides_buffer,
                                                                        depth_buffer,depthshape_buffer,depthstrides_buffer,
                                                                        
                                                                        shape_buffer,
                                                                        axissumflag_buffer,
                                                                        sumstrides_buffer,
                                                                        kz,ky,kx,rho,cp,
                                                                        result_buffer,resultshape_buffer,resultstrides_buffer,
                                                                        
                                                                        ndim,
                                                                        wait_for=(fill_event,))
        pass
    else:
        # opencl_greensfcn_curved kernel
        kernel_event=greensconvolution_params.opencl_greensfcn_curved_function(
            queue,(iterlen,),None,
            linelength_buffer,linelengthshape_buffer,linelengthstrides_buffer,
            tvec_buffer,tvecshape_buffer,tvecstrides_buffer,
            source_intensity_buffer,source_intensityshape_buffer,source_intensitystrides_buffer,
            depth_buffer,depthshape_buffer,depthstrides_buffer,
            theta_buffer,thetashape_buffer,thetastrides_buffer,
            avgcurvatures_buffer,avgcurvaturesshape_buffer,avgcurvaturesstrides_buffer,
            avgcrosscurvatures_buffer,avgcrosscurvaturesshape_buffer,avgcrosscurvaturesstrides_buffer,
            iop_dy_buffer,iop_dy_shape_buffer,iop_dy_strides_buffer,
            iop_dx_buffer,iop_dx_shape_buffer,iop_dx_strides_buffer,
            shape_buffer,axissumflag_buffer,sumstrides_buffer,
            kz,ky,kx,rho,cp,
            result_buffer,resultshape_buffer,resultstrides_buffer,
            ndim,
            wait_for=(fill_event,))
        pass
    copyout_event=cl.enqueue_copy(queue,result,result_buffer,wait_for=(kernel_event,),is_blocking=False)

    
    resultref=returnparam(result)  # python (not cython) reference that we can access from closure
    resultsummedshaperef=returnparam(resultsummedshape) #returnparam(resultshape)

    # Closure function is what we have to do to clean up. and get data
    def closure():
        copyout_event.wait()
        
        
        linelength_buffer.release()
        linelengthshape_buffer.release()
        linelengthstrides_buffer.release()
        
        tvec_buffer.release()
        tvecshape_buffer.release()
        tvecstrides_buffer.release()
        
        source_intensity_buffer.release()
        source_intensityshape_buffer.release()
        source_intensitystrides_buffer.release()

        depth_buffer.release()
        depthshape_buffer.release()
        depthstrides_buffer.release()

        theta_buffer.release()
        thetashape_buffer.release()
        thetastrides_buffer.release()
        
        if avgcurvatures is not None:
            avgcurvatures_buffer.release()
            avgcurvaturesshape_buffer.release()
            avgcurvaturesstrides_buffer.release()

            avgcrosscurvatures_buffer.release()
            avgcrosscurvaturesshape_buffer.release()
            avgcrosscurvaturesstrides_buffer.release()
            pass

        if iop_dy is not None:
            iop_dy_buffer.release()
            iop_dy_shape_buffer.release()
            iop_dy_strides_buffer.release()

            iop_dx_buffer.release()
            iop_dx_shape_buffer.release()
            iop_dx_strides_buffer.release()
            pass
            

        shape_buffer.release()
        axissumflag_buffer.release()
        resultshape_buffer.release()

        result_buffer.release()
        resultstrides_buffer.release()
        
        
        
        return resultref.reshape(*resultsummedshaperef)
            
    if opencl_queue is not None:
        # queue parameter specified... return closure

        return closure
    queue.finish()
    # Otherwise, execute closure, freeing memory immediately and performing summation
    return closure()
    pass




def returnparam(param): # Work around Cython's inability to use a cdef'd numpy array in a local function
    return param

# NOTE: The old version automatically summed over size of zvec which was same as rvec, and vectorized or tvec, and coeff=1.0. So
# to convert, set coeff=1.0, reshape zvec, rvec to be zvec.shape[0] x tvec.shape[0], set sumaxes to (0,)
#@cython.boundscheck(False)
def greensconvolution_integrate_anisotropic(greensconvolution_params,zvecinput,xvecinput,tvecinput,yval,kz,ky,kx,rho,cp,coeff,sumaxes,avgcurvatures=None,kernel=default_kernel,opencl_queue=None):
    # zvec, rvec, and tvec should have the same number of dimensions, and
    # this function broadcasts over singleton dimensions and then
    # sums over the axes specified in sumaxes.

    # zvec is depths from surface (initial propagation)
    # rconductivityscaledvec is scaled distance of measurement from scatterer (scattered propagation).
    # rconductivityscaledvec should be sqrt(x^2*(kz/kx) + y^2*(kz/ky)+z^2)
    # tvec may have a different length and the result will be an array

    # If opencl_queue is not None, then this function will run in that queue and returns a closure instead of the result array.
    # Call the closure to wait for the operation to complete and obtain the result array

    # if avgcurvatures is set then kernel must support curvature

    
    # !!!**** Should automatically determine from GPU max memory allocation
    # Whether to perform sum ourselves with CPU or to have kernel do it !!!***

    cdef np.ndarray[float,mode="c"] zvec=zvecinput.reshape(np.prod(zvecinput.shape))
    cdef np.ndarray[float,mode="c"] xvec=xvecinput.reshape(np.prod(xvecinput.shape))
    cdef np.ndarray[float,mode="c"] tvec=tvecinput.reshape(np.prod(tvecinput.shape))
    cdef np.ndarray[float,mode="c"] avgcurvatures_cython
    
    cdef np.ndarray[int64_t,ndim=1,mode="c"] zvecshape
    cdef np.ndarray[int64_t,ndim=1,mode="c"] tvecshape 
    cdef np.ndarray[int64_t,ndim=1,mode="c"] xvecshape
    cdef np.ndarray[int64_t,ndim=1,mode="c"] avgcurvaturesshape
    cdef np.ndarray[int64_t,ndim=1,mode="c"] axissumflag
    cdef np.ndarray[int64_t,ndim=1,mode="c"] shape
    #cdef np.ndarray[int64_t,ndim=1,mode="c"] iterationstrides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] sumstrides


    cdef np.ndarray[int64_t,ndim=1,mode="c"] zvecstrides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] tvecstrides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] xvecstrides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] avgcurvaturesstrides
    cdef np.ndarray[int64_t,ndim=1,mode="c"] resultstrides

    cdef uint64_t ndim=zvecinput.ndim

    if ndim != tvecinput.ndim or ndim != xvecinput.ndim:
        raise ValueError("ndim mismatch: zvec: %d tvec: %d xvec: %d" % (zvecinput.ndim,tvecinput.ndim,xvecinput.ndim))
    

    if avgcurvatures is not None and not "_curved" in kernel:
        raise ValueError("Calculation with curvatures requires curved kernel")
    elif "_curved" in kernel and avgcurvatures is None:
        raise ValueError("Calculation with curvature kernel requires avgcurvatures parameter")
    
    
    zvecshape=np.array(zvecinput.shape,dtype=np.int64)
    zvecstrides=np.array(np.cumprod(np.concatenate((zvecshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)

    if avgcurvatures is not None and (zvec < 0.0).any():
        raise ValueError("curvature case only validated for z > 0")
    
    tvecshape=np.array(tvecinput.shape,dtype=np.int64)
    tvecstrides=np.array(np.cumprod(np.concatenate((tvecshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)
    xvecshape=np.array(xvecinput.shape,dtype=np.int64)
    xvecstrides=np.array(np.cumprod(np.concatenate((xvecshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)
    
    if avgcurvatures is not None:
        avgcurvatures_cython=avgcurvatures.reshape(np.prod(avgcurvatures.shape))
        avgcurvaturesshape=np.array(avgcurvatures.shape,dtype=np.int64)
        avgcurvaturesstrides=np.array(np.cumprod(np.concatenate((avgcurvaturesshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)
        pass

    shape=np.array((zvecshape,tvecshape,xvecshape),dtype=np.int64).max(0)
    shapeones=np.ones(ndim,dtype=np.int64)

    #print("shape=%s" % (str(shape)))
    #print(zvecshape)
    #print(tvecshape)
    #print(rvecshape)

    if (not(((shape==zvecshape) | (zvecshape==shapeones)).all()) or
        not(((shape==tvecshape) | (tvecshape==shapeones)).all()) or
        not(((shape==xvecshape) | (xvecshape==shapeones)).all())):
        raise ValueError("Non-broadcastable shape mismatch. zvec=%s, tvec=%s, xvec=%s shape=%s" % (str(zvecshape),str(tvecshape),str(xvecshape),str(shape)))
    
    
    
    # iterationstrides=np.array(np.cumprod(np.concatenate((iterationshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)

    if sumaxes is None:
        sumaxes = ()
        pass


    Onboard_summation_OK=True
    
    if kernel.startswith("opencl_"):
        if cl is None:
            raise ValueError("Exception importing pyopencl (pyopencl is required for OpenCL support)")

        greensconvolution_params.get_opencl_context()

        # Inhibit QUADPACK kernel from performing summation 
        # on AMD GPU's as this is known (in at least some cases)
        # to give erroneous results, probably due to irreducable 
        # control flow that is not rejected as an error by AMD's 
        # library
        
        # It seems to be OK so long as we don't loop in the kernel 
        # beyond what is already in QUADPACK
        # 
        # ***!!! This is probably fixed now that a barrier has been added around the looping, but need to confirm !!!***
        if any([ dev.vendor_id==4098 and (dev.type & cl.device_type.GPU) for dev in greensconvolution_params.OpenCL_CTX.devices ]) and kernel=="opencl_quadpack":
            Onboard_Summation_OK=False
            print("greensconvolution: GPU Onboard summation disabled for AMD GPU and opencl_quadpack kernel")
            pass
        pass
	
        max_alloc_size=np.inf
        for dev in greensconvolution_params.OpenCL_CTX.devices:
            if dev.max_mem_alloc_size != 0 and dev.max_mem_alloc_size < max_alloc_size:
                max_alloc_size=dev.max_mem_alloc_size
                pass
            if hasattr(dev,"max_global_variable_size") and dev.max_global_variable_size != 0 and dev.max_global_variable_size < max_alloc_size:
                max_alloc_size=dev.max_global_variable_size
                pass
                
            pass
        pass
    
    if kernel.startswith("opencl_") and np.prod(shape)*sizeof(float) + np.prod(zvecshape)*sizeof(float) + np.prod(tvecshape)*sizeof(float) + np.prod(xvecshape)*sizeof(float) > 0.8*max_alloc_size  and Onboard_Summation_OK:
        print("greensconvolution: Performing onboard summation on GPU to avoid exceeding GPU memory")
        
        CSumAxes=sumaxes 
        PySumAxes=()
        pass
    else: 
        CSumAxes=() # Sum axes in Python by default
        PySumAxes=sumaxes
        
        pass
    axissumflag=np.zeros(ndim,dtype=np.int64)
    if len(CSumAxes) > 0:
        #print(sumaxes)
        #print(axissumflag)
        axissumflag[np.array(CSumAxes)]=1  # Numpy treats an array as a single index, whereas a tuple would be a sequence of indices, which isn't what we want here.
        pass
        
    PyAxisSumFlag=np.zeros(ndim,dtype=np.int64)
    if len(PySumAxes) > 0:
        #print(sumaxes)
        #print(axissumflag)
        PyAxisSumFlag[np.array(PySumAxes)]=1  # Numpy treats an array as a single index, whereas a tuple would be a sequence of indices, which isn't what we want here.
        pass
        


    sumshape = np.array(shape,dtype=np.int64)
    sumshape[~axissumflag.astype(np.bool)]=1
    sumstrides=np.array(np.cumprod(np.concatenate((sumshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)
    #print("sumshape=%s ; sumstrides=%s" % (str(sumshape),str(sumstrides)))
    #sys.stderr.write("iterationstrides=%s dtype=%s\n" % (str(iterationstrides),str(iterationstrides.dtype)))
    resultshape=np.array(shape,dtype=np.int64)
    resultshape[axissumflag.astype(np.bool)]=1
    resultstrides = np.array(np.cumprod(np.concatenate((resultshape,(1,)))[-1:0:-1])[::-1],dtype=np.int64)
    
    
    cdef np.ndarray[float,ndim=2,mode="c"] vrange
    cdef float *vrange_c
    
    cdef np.ndarray[float,ndim=2,mode="c"] crange
    cdef float *crange_c
    
    cdef np.ndarray[float,ndim=2,mode="c"] integraleval
    cdef float *integraleval_c

    cdef np.ndarray[float,ndim=2,mode="c"] integral_dintegranddveval
    cdef float *integral_dintegranddveval_c
    
    cdef np.ndarray[float,ndim=2,mode="c"] integral_dintegranddceval
    cdef float *integral_dintegranddceval_c

    cdef uint64_t count
    cdef double log10v0,log10c0,dlog10v,dlog10c,vval,cval
    cdef int64_t vidx,vidx2,cidx,cidx2
    cdef int64_t point_vidx[4]
    cdef int64_t point_cidx[4]
    cdef double vidxval,cidxval,integralevalpt,integral_dintegranddvevalpt,integral_dintegranddcevalpt
    cdef double totalweight,est
    cdef int pointcnt
    cdef double accum
    cdef np.ndarray[float,mode="c"] result
    cdef float *result_c
    cdef int64_t tcnt,nt
    cdef int64_t iterlen
    cdef double alpha
     
    cdef float *zvec_c
    cdef float *xvec_c
    cdef float *tvec_c

    cdef np.ndarray[float,ndim=1,mode="c"] float_zero=np.array((0.0,),dtype='f')
    

    cdef float alphaz=kz*1.0/(rho*cp)
    assert(kx==ky)
    cdef float alphaxy=kx*1.0/(rho*cp)
    
    cdef float alphaxyz=((kx*ky*kz)**(1.0/3.0))/(rho*cp)
    
    cdef float coeff_prod = coeff*2.0*(alphaz**(3.0/2.0))/((rho*cp*pi*pi)*(alphaxyz**(3.0/2.0)))
    

    
    nvrange=greensconvolution_params.vrange.shape[0]
    ncrange=greensconvolution_params.crange.shape[1]

    
    if kernel=="openmp_interpolator":

        resultactualshape=shape[~axissumflag.astype(np.bool)]
        result=np.zeros(np.prod(resultactualshape),dtype='f')

        vrange=greensconvolution_params.vrange
        crange=greensconvolution_params.crange
        integraleval=greensconvolution_params.integraleval
        integral_dintegranddveval=greensconvolution_params.integral_dintegranddveval
        integral_dintegranddceval=greensconvolution_params.integral_dintegranddceval
        
        vrange_c = <float *> np.PyArray_DATA(vrange) 
        crange_c = <float *> np.PyArray_DATA(crange) 
        integraleval_c = <float *> np.PyArray_DATA(integraleval) 
        integral_dintegranddveval_c = <float *> np.PyArray_DATA(integral_dintegranddveval) 
        integral_dintegranddceval_c = <float *> np.PyArray_DATA(integral_dintegranddceval) 
        
        
        zvec_c = <float *> np.PyArray_DATA(zvec) 
        xvec_c = <float *>np.PyArray_DATA(xvec)
        tvec_c = <float *> np.PyArray_DATA(tvec)
        
        
        
        
        assert(vrange.shape[1]==1)
        assert(crange.shape[0]==1)

    

    
        #print(axissumflag.astype(np.bool))
        #print(shape)
        #print(shape[~axissumflag.astype(np.bool)])
        result_c = <float *> np.PyArray_DATA(result) 
        
        #sys.stderr.write("strides=%s dtype=%s\n" % (str(strides),str(strides.dtype)))
        
        greensconvolution_integrate_anisotropic_c(
            vrange_c,nvrange, # number of rows in integraleval
            crange_c,ncrange, # number of cols in integraleval
            integraleval_c,
            integral_dintegranddveval_c, # same size as integraleval
            integral_dintegranddceval_c, # same size as integraleval
            zvec_c, <uint64_t *>np.PyArray_DATA(zvecshape),<uint64_t *>np.PyArray_DATA(zvecstrides),
            xvec_c, <uint64_t *>np.PyArray_DATA(xvecshape),<uint64_t *>np.PyArray_DATA(xvecstrides),
            tvec_c, <uint64_t *>np.PyArray_DATA(tvecshape),<uint64_t *>np.PyArray_DATA(tvecstrides),
            yval,
            NULL,NULL,NULL,
            <uint64_t *>np.PyArray_DATA(sumstrides),<uint64_t *>np.PyArray_DATA(shape),
            alphaz,alphaxy,0,
            result_c,<uint64_t *>np.PyArray_DATA(resultstrides),
            coeff_prod,
            <uint64_t *>np.PyArray_DATA(axissumflag),
            ndim)
        pass
    elif kernel.startswith("opencl_"):
        if cl is None:
            raise ValueError("Exception importing pyopencl (pyopencl is required for OpenCL support)")

        # greensconvolution_params.get_opencl_context() # now done earlier

        # print(shape)
        # print(axissumflag)
        resultactualshape=shape[~axissumflag.astype(np.bool)]
        # print(resultactualshape)
        if greensconvolution_params.OpenCL_Version < 1.2:
            # Cannot use fill... copy zero buffer instead
            result=np.zeros(np.prod(resultactualshape),dtype='f')
            pass
        else:
            result=np.empty(np.prod(resultactualshape),dtype='f')
            pass
            
            
        
        queue=opencl_queue
        if queue is None:
            queue=cl.CommandQueue(greensconvolution_params.OpenCL_CTX,properties=greensconvolution.greensconvolution_calc.OpenCL_GetOutOfOrderDeviceQueueProperties(greensconvolution_params.OpenCL_CTX))
            pass
        assert(queue is not None)
            
        zvec_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=zvec)
        zvecshape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=zvecshape)
        zvecstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=zvecstrides)
        
        xvec_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=xvec)
        xvecshape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=xvecshape)
        xvecstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=xvecstrides)
        
        tvec_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=tvec)
        tvecshape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=tvecshape)
        tvecstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=tvecstrides)

        if avgcurvatures is not None:
            avgcurvatures_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=avgcurvatures_cython)
            avgcurvaturesshape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=avgcurvaturesshape)
            avgcurvaturesstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=avgcurvaturesstrides)
            pass
        pass
            
        shape_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=shape)
        sumstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=sumstrides)

        #print("result.nbytes=%fG" % (result.nbytes/1.e9))
        result_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_WRITE,size=result.nbytes)
        #print("allocated result buffer.")
        resultstrides_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=resultstrides)
        #sys.stderr.write("allocated resultstrides buffer.\n")

            
        axissumflag_buffer=cl.Buffer(greensconvolution_params.OpenCL_CTX,cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,hostbuf=axissumflag)
        #print("allocated axissumflag buffer.")
        LoadGCKernel(greensconvolution_params, kernel)
        #sys.stderr.write("Loaded GC kernel.\n")
        # fill buffer with zeros
        iterlen=np.prod(resultshape)
        #print("iterlen=%fG" % (iterlen/1.e9))
        
        if greensconvolution_params.OpenCL_Version < 1.2:
            fill_event=cl.enqueue_copy(queue,result_buffer,result,is_blocking=False);
            pass
        else:
            fill_event=cl.enqueue_fill_buffer(queue,result_buffer,float_zero,0,result.nbytes);
            pass

        sys.stderr.flush()
        sys.stdout.flush()
        # MUST HAVE KERNEL WAIT FOR FILL EVENT TO COMPLETE!!!

        kernel_event=None
        if kernel=="opencl_interpolator":
            kernel_event=greensconvolution_params.opencl_interpolator_function(queue,(iterlen,),None,
                                                                               greensconvolution_params.opencl_interpolator_vrange_buffer,nvrange,
                                                                               greensconvolution_params.opencl_interpolator_crange_buffer,ncrange,
                                                                               greensconvolution_params.opencl_interpolator_integraleval_buffer,
                                                                               greensconvolution_params.opencl_interpolator_integral_dintegranddveval_buffer,
                                                                               greensconvolution_params.opencl_interpolator_integral_dintegranddceval_buffer,
                                                                               zvec_buffer,zvecshape_buffer,zvecstrides_buffer,
                                                                               xvec_buffer,xvecshape_buffer,xvecstrides_buffer,
                                                                               tvec_buffer,tvecshape_buffer,tvecstrides_buffer,
                                                                               yval,
                                                                               sumstrides_buffer,
                                                                               shape_buffer,
                                                                               alphaz,alphaxy,
                                                                               result_buffer,resultstrides_buffer,
                                                                               coeff_prod,
                                                                               axissumflag_buffer,
                                                                               ndim,
                                                                               wait_for=(fill_event,))
                
            pass
        elif kernel=="opencl_interpolator_curved":
            kernel_event=greensconvolution_params.opencl_interpolator_curved_function(queue,(iterlen,),None,
                                                                                      greensconvolution_params.opencl_interpolator_vrange_buffer,nvrange,
                                                                                      greensconvolution_params.opencl_interpolator_crange_buffer,ncrange,
                                                                                      greensconvolution_params.opencl_interpolator_integraleval_buffer,
                                                                                      greensconvolution_params.opencl_interpolator_integral_dintegranddveval_buffer,
                                                                                      greensconvolution_params.opencl_interpolator_integral_dintegranddceval_buffer,
                                                                                      zvec_buffer,zvecshape_buffer,zvecstrides_buffer,
                                                                                      xvec_buffer,xvecshape_buffer,xvecstrides_buffer,
                                                                                      tvec_buffer,tvecshape_buffer,tvecstrides_buffer,
                                                                                      yval,
                                                                                      avgcurvatures_buffer,avgcurvaturesshape_buffer,avgcurvaturesstrides_buffer,
                                                                                      sumstrides_buffer,
                                                                                      shape_buffer,
                                                                                      alphaz,alphaxy,
                                                                                      result_buffer,resultstrides_buffer,
                                                                                      coeff_prod,
                                                                                      axissumflag_buffer,
                                                                                      ndim,
                                                                                      wait_for=(fill_event,))
            
            pass
        elif kernel=="opencl_simplegaussquad":
            kernel_event=greensconvolution_params.opencl_simplegaussquad_function(queue,(iterlen,),None,
                                                                                  greensconvolution_params.opencl_simplegaussquad_points_buffer,
                                                                                  greensconvolution_params.opencl_simplegaussquad_weights_buffer,
                                                                                  zvec_buffer,zvecshape_buffer,zvecstrides_buffer,
                                                                                  xvec_buffer,xvecshape_buffer,xvecstrides_buffer,
                                                                                  tvec_buffer,tvecshape_buffer,tvecstrides_buffer,
                                                                                  yval,
                                                                                  sumstrides_buffer,shape_buffer,
                                                                                  alphaz,alphaxy,
                                                                                  result_buffer,resultstrides_buffer,
                                                                                  coeff_prod,
                                                                                  axissumflag_buffer,
                                                                                  ndim,
                                                                                  wait_for=(fill_event,))
            
            pass
        elif kernel=="opencl_quadpack":
            kernel_event=greensconvolution_params.opencl_quadpack_function(queue,(iterlen,),None,
                                                                           zvec_buffer,zvecshape_buffer,zvecstrides_buffer,
                                                                           xvec_buffer,xvecshape_buffer,xvecstrides_buffer,
                                                                           tvec_buffer,tvecshape_buffer,tvecstrides_buffer,
                                                                           yval,
                                                                           sumstrides_buffer,shape_buffer,
                                                                           alphaz,
                                                                           alphaxy,
                                                                           result_buffer,resultstrides_buffer,
                                                                           coeff_prod,
                                                                           axissumflag_buffer,
                                                                           ndim,
                                                                           wait_for=(fill_event,))
                
            pass
        else:
            raise ValueError("Unknown opencl kernel %s" % (kernel))
            
        # copy data out
        copyout_event=cl.enqueue_copy(queue,result,result_buffer,wait_for=(kernel_event,),is_blocking=False)
            

        resultref=returnparam(result)  # python (not cython) reference that we can access from closure
        shaperef=returnparam(shape)
        PySummedShape=shape[(~axissumflag.astype(np.bool)) & (~PyAxisSumFlag.astype(np.bool))]

        # Closure function is what we have to do to clean up. and get data
        def closure():
            copyout_event.wait()
            
            
            zvec_buffer.release()
            zvecshape_buffer.release()
            zvecstrides_buffer.release()
                
            xvec_buffer.release()
            xvecshape_buffer.release()
            xvecstrides_buffer.release()
            
            tvec_buffer.release()
            tvecshape_buffer.release()
            tvecstrides_buffer.release()
            
                
            # iterationshape_buffer.release()
            
            sumstrides_buffer.release()
            shape_buffer.release()
            
            result_buffer.release()
            resultstrides_buffer.release()
            axissumflag_buffer.release()
            
            

            if len(PySumAxes) > 0:
                return resultref.reshape(*shaperef).sum(PySumAxes).reshape(*PySummedShape)
            else:
                return resultref.reshape(*PySummedShape)
            pass
                  
        if opencl_queue is not None:
            # queue parameter specified... return closure

            return closure
        queue.finish()
        # Otherwise, execute closure, freeing memory immediately and performing summation
        return closure()
        pass
    else:
        raise ValueError("Unknown kernel %s" % (kernel))
    
        
    
    PySummedShape=shape[(~axissumflag.astype(np.bool)) & (~PyAxisSumFlag.astype(np.bool))]

    #print(PySumAxes)
    #print(resultshape)
    #print(PySummedShape)
    #print(result.sum(PySumAxes).shape)

    if len(PySumAxes) > 0:
        return result.reshape(*shape).sum(PySumAxes).reshape(*PySummedShape)
    else:
        return result.reshape(*PySummedShape)
    pass




notused=r"""  # This function turned out not to be needed after all
def greensconvolution_calcarray(greensconvolution_params,np.ndarray[double,ndim=1] zvec,np.ndarray[double,ndim=1] rvec,np.ndarray[double,ndim=1] tvec,k,rho,cp):
    # Like greensconvolution_integrate, but we return a multidimensional
    # array rather than integrating over zvec and rvec. 
    # zvec and rvec should have the same length
    # tvec may have a different length and the result will be an array

    cdef np.ndarray[double,ndim=2] vrange
    cdef np.ndarray[double,ndim=2] crange
    cdef np.ndarray[double,ndim=2] integraleval
    cdef np.ndarray[double,ndim=2] integral_dintegranddveval
    cdef np.ndarray[double,ndim=2] integral_dintegranddceval

    cdef np.ndarray[double,ndim=1] coeff
    cdef np.ndarray[double,ndim=1] v
    cdef np.ndarray[double,ndim=1] c
    cdef uint64_t count
    cdef double log10v0,log10c0,dlog10v,dlog10c,vval,cval
    cdef int64_t vidx,vidx2,cidx,cidx2
    cdef int64_t point_vidx[4]
    cdef int64_t point_cidx[4]
    cdef double vidxval,cidxval,integralevalpt,integral_dintegranddvevalpt,integral_dintegranddcevalpt
    cdef double vals[4],
    cdef double weights[4]
    cdef double totalweight,est
    cdef int pointcnt
    cdef double accum
    cdef np.ndarray[double,ndim=2] result
    cdef int64_t tcnt,nt
    cdef double alpha
    
    (vrange,crange,integraleval,integral_dintegranddveval,integral_dintegranddceval,OpenCL_CTX)=greensconvolution_params
    
    alpha=k*1.0/(rho*cp)

    coeff=2.0/((rho*cp*pi*pi)*zvec**3.0)

    nvrange=vrange.shape[0]
    ncrange=crange.shape[1]
    assert(vrange.shape[1]==1)
    assert(crange.shape[0]==1)

    
    log10v0=log10(vrange[0,0])
    log10c0=log10(crange[0,0])
    dlog10v=log10(vrange[1,0])-log10(vrange[0,0])
    dlog10c=log10(crange[0,1])-log10(crange[0,0])

    nz=zvec.shape[0]
    assert(nz==rvec.shape[0])
    
    nt=tvec.shape[0]
    result=np.zeros((nz,nt),dtype='d')
    
    c=np.abs(rvec/zvec)
    for tcnt in range(nt):

        assert(tvec[tcnt] > 0)
        
        v=(4*alpha*tvec[tcnt])/zvec**2.0
        
        
    

        for count in range(nz):
            vval=v[count]            
            cval=c[count]

            # print("%f, %f, %f" % (log10(vval),log10v0,dlog10v))
            
            vidx=int((log10(vval)-log10v0)/dlog10v)
            # print("vidx=%d; nvrange=%d" % (vidx,nvrange))
            assert(vidx >= 0 and vidx+1 < nvrange) 
            vidx2=vidx+1
            
            
            cidx=int((log10(cval)-log10c0)/dlog10c)
            # print("cidx=%d; ncrange=%d" % (cidx,ncrange))
            assert(cidx >= 0 and cidx+1 < ncrange) 
            cidx2=cidx+1
            
            point_vidx[0]=vidx
            point_cidx[0]=cidx
            
            point_vidx[1]=vidx
            point_cidx[1]=cidx2

            point_vidx[2]=vidx2
            point_cidx[2]=cidx

            point_vidx[3]=vidx2
            point_cidx[3]=cidx2

            totalweight=0.0
        
            for pointcnt in range(4):
                vidxval=vrange[point_vidx[pointcnt],0]
                cidxval=crange[0,point_cidx[pointcnt]]
            
                integralevalpt=integraleval[point_vidx[pointcnt],point_cidx[pointcnt]]
                integral_dintegranddvevalpt=integral_dintegranddveval[point_vidx[pointcnt],point_cidx[pointcnt]]
                integral_dintegranddcevalpt=integral_dintegranddceval[point_vidx[pointcnt],point_cidx[pointcnt]]
                vals[pointcnt] = integralevalpt + (vval-vidxval)*integral_dintegranddvevalpt +(cval-cidxval)*integral_dintegranddcevalpt;
                weights[pointcnt]=sqrt(1.0/(0.001+(vval-vidxval)*(vval-vidxval) + (cval-cidxval)*(cval-cidxval)))
                totalweight+=weights[pointcnt]
            
                pass
            
            est=0.0
            for pointcnt in range(4):
                est+=vals[pointcnt]*weights[pointcnt]/totalweight
                pass

            # Limit according to nonnegative and upper bound in greensfcn_doc.tex
            if est < 0.0:
                # print("Warning: Integral gave inaccurate calculation of %g at v=%g, c=%g; lower bound of 0 used instead" % (est,vval,cval),file=sys.stderr)
                est=0.0
                pass
            elif est > 0.185*exp(-(cval**2.0-1.0)/vval):
                print("Warning: Integral gave inaccurate calculation of %g at v=%g,c=%g; upper bound of %g used instead" % (est,vval,cval,0.185*exp(-(cval**2.0-1.0)/vval)),file=sys.stderr)
                est= 0.185*exp(-(cval**2.0-1.0)/vval)
                pass
            
            result[count,tcnt]=coeff[count]*est
            pass
        pass
    return result
"""
