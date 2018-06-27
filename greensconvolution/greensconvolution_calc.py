# see greensfcn_doc.tex/.pdf
#
#
# Expressions from greensfcn_deriv.wxml

import re
import numpy as np
import scipy as sp
import scipy.integrate
from scipy.integrate import quad
from netCDF4 import Dataset
import os
import os.path

try:
    import pyopencl as cl
    pass
except:
    cl=None
    pass

try: 
    basestring # Will error out in python3
    pass
except NameError:
    basestring=str
    pass


def OpenCL_GetOutOfOrderDeviceQueueProperties(Context):
    # Determine queue properties to get an out-of-order device queue in the 
    # given context, if possible
    props=0

    QueuePropsList=[ dev.get_info(cl.device_info.QUEUE_PROPERTIES) for dev in Context.devices ]
    
    # And all properties together to find common props
    QueuePropsCommon=np.bitwise_and.reduce(np.array(QueuePropsList,dtype=np.uint32))
    
    if cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE & QueuePropsCommon:
        # if all devices support out-of-order execution
        props |= cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
        pass
    
    versions=[ re.match(r"""OpenCL (\d+)[.](\d+) """,dev.version).groups() for dev in Context.devices ] # list of ('major version', 'minor version') tuples

    versionsnumeric = [ (int(major),int(minor)) for (major,minor) in versions ]
    
    minver = min(versionsnumeric)

    #if minver >= (2,0):
    #    # OpenCL2.0 supports cl.command_queue_properties.ON_DEVICE
    #    (I think we have to query this capability explicitly (?)
    #    props |= cl.command_queue_properties.ON_DEVICE
    #    pass
    
    return props

def evalfrompoint(vpt,cpt,integral_evalpt,integral_dintegranddvevalpt,integral_dintegranddcevalpt,v,c):
    return integral_evalpt + (v-vpt)*integral_dintegranddvevalpt + (c-cpt)*integral_dintegranddcevalpt

def evalfromneighbors(vrange,crange,integraleval,integral_dintegranddveval,integral_dintegranddceval,v,c,verbose=False):
    vidx=np.argmin(abs(vrange-v))
    if vrange[vidx,0] > v:
        vidx2=vidx-1
        pass
    else:
        vidx2=vidx+1
        pass
    
    cidx=np.argmin(abs(crange-c))
    if crange[0,cidx] > c:
        cidx2=cidx-1
        pass
    else:
        cidx2=cidx+1
        pass

    points=[(vidx,cidx),(vidx,cidx2),(vidx2,cidx),(vidx2,cidx2)]

    vals=[]
    weights=[]
    for (vi,ci) in points:
        val=evalfrompoint(vrange[vi,0],crange[0,ci],integraleval[vi,ci],integral_dintegranddveval[vi,ci],integral_dintegranddceval[vi,ci],v,c)
        weight=np.sqrt(1.0/(0.001+(vrange[vi,0]-v)**2.0 + (crange[0,ci]-c)**2.0))
        if verbose:
            print('Projection from %f@(%f,%f): %f' % (integraleval[vi,ci],vrange[vi,0],crange[0,ci],val))
            pass
        vals.append(val)
        weights.append(weight)
        pass

    vals=np.array(vals,dtype='d')
    weights=np.array(weights,dtype='d')
    est=np.max((0,np.dot(vals,weights/np.sum(weights))))
    if verbose:
        print('Estimated value at (%f,%f): %f' % (v,c,est))
        print('Actual value: %f' % (integral(v,c)[0]))
        print(' ')
        pass
    return est

class greensconvolution_params(object):
    vrange=None  # Note: these buffers relied upon as buffer backing  by opencl_interpolator
    crange=None
    integraleval=None
    integral_dintegranddveval=None
    integral_dintegranddceval=None

    OpenCL_CTX=None
    OpenCL_Platform=None
    OpenCL_Version=None # filled out by figure_out_version() once OpenCL_Platform is set
    
    # opencl programs:
    opencl_interpolator=None # The cl.Program
    opencl_interpolator_function=None # The function to call
    opencl_interpolator_curved_function=None # The function to call
    opencl_interpolator_vrange_buffer=None
    opencl_interpolator_crange_buffer=None
    opencl_interpolator_integraleval_buffer=None
    opencl_interpolator_integral_dintegranddveval=None
    opencl_interpolator_integral_dintegranddceval=None
    
    opencl_simplegaussquad=None # The cl.Program
    opencl_simplegaussquad_function=None
    opencl_simplegaussquad_points=None  # Points array
    opencl_simplegaussquad_points_buffer=None  # Points buffer
    opencl_simplegaussquad_weights=None  # Weights array
    opencl_simplegaussquad_weights_buffer=None  # Weights buffer
    
    opencl_quadpack=None
    
    opencl_imagesources=None
    opencl_imagesources_function=None

    opencl_imagesources_curved=None
    opencl_imagesources_curved_function=None

    opencl_greensfcn_curved=None
    opencl_greensfcn_curved_function=None
    
    def __init__(self,**kwargs):
        for kwarg in kwargs:
            if not hasattr(self,kwarg):
                raise AttributeError("Unknown attribute %s" % (kwarg))
        
            setattr(self,kwarg,kwargs[kwarg])
            pass
        pass

    def figure_out_version(self):
        # figure out version info given platform and device

        #device_version=float(re.match(r"""OpenCL (\d+[.]\d+) """,self.OpenCL_Platform.device_version).group(1))
        
        platform_version=float(re.match(r"""OpenCL (\d+[.]\d+) """,self.OpenCL_Platform.version).group(1))
        
        #self.OpenCL_Version=min(device_version,platform_version)
        self.OpenCL_Version=platform_version
        pass

    def get_opencl_context(self,device_type_requested=None,device_name=None):
        # Can set device_type to_requested a string, e.g. GPU, Accelerator, CPU, etc.
        if isinstance(device_type_requested,basestring):
            device_types=[getattr(cl.device_type,device_type_requested)]
            pass
        elif device_type_requested is None:
            device_types=[cl.device_type.GPU,cl.device_type.CPU]
            pass
        else: 
            device_types=[ device_type ]
            pass

        if cl is None:
            raise ValueError("Exception importing pyopencl (pyopencl is required for OpenCL support)")

        found_device=False
        if self.OpenCL_CTX is None:
            for device_type in device_types: 
                platforms = cl.get_platforms()
                if found_device:
                    break

                for platform in platforms:
                    if found_device:
                        break
                    devices=platform.get_devices(device_type=device_type)
                    for device in devices:
                        if found_device:
                            break
                        if device.type & device_type:
                            if device_name is None or device_name==device.name:
                                self.OpenCL_CTX = cl.Context(
                                    devices=[device],
                                    properties=[(cl.context_properties.PLATFORM, platform)])
                                found_device=True
                                self.OpenCL_Platform=platform
                                self.figure_out_version()
                                pass
                            pass
                        pass
                            
                    pass
                pass
            pass
        if self.OpenCL_CTX is None:
            raise Exception("Failed to find a suitable OpenCL context")
        return self.OpenCL_CTX
    
    pass
    


def read_greensconvolution(filename=None):
    # filename is path to greensconvolution.nc file, which
    # is a NetCDF4 store of pre-calculated evaluations of
    # the integral, generated by running this python file as a script.
    # if filename is not specified, then it looks for it in the
    # same directory as this file. 
    
    if filename is None:
        filename=os.path.join(os.path.dirname(__file__),"greensconvolution.nc")
        pass
    
    
    rootgrp=Dataset(filename,"r")
    
    vrange=rootgrp.variables["v"][::]
    vrange=vrange.reshape((vrange.shape[0],1))
    crange=rootgrp.variables["c"][::]
    crange=crange.reshape((1,crange.shape[0]))
    integraleval=rootgrp.variables["integraleval"][::]
    integral_dintegranddveval=rootgrp.variables["integral_dintegranddveval"][::]
    integral_dintegranddceval=rootgrp.variables["integral_dintegranddceval"][::]

    rootgrp.close()

    
    return greensconvolution_params(vrange=vrange,
                                    crange=crange,
                                    integraleval=integraleval,
                                    integral_dintegranddveval=integral_dintegranddveval,
                                    integral_dintegranddceval=integral_dintegranddceval)

integrand=lambda u,v,c: u**((-3.0)/2.0)*(v-u)**((-3.0)/2.0)*np.exp((-c**2/(v-u))-1/u)

# Rewrite integrand for integration from zero to 1, so we can
# differentiate it better w.r.t. u
# i.e. let w be u/v -> u = vw -> dw = du/v -> du = v*dw
#   (vw)**((-3.0)/2.0)*(v-vw)**((-3.0)/2.0)*np.exp((-c**2/(v-vw))-1/(vw)) v*dw
# this is integrated as w=0..1
#   = v**(-3.0) * w**((-3.0)/2.0) * (1-w)**((-3.0)/2.0) * np.exp((1/v)* ((-c**2/(1-w))-1/(w))) v*dw
# this is integrated as w=0..1
#   = v**(-2.0) * w**((-3.0)/2.0) * (1-w)**((-3.0)/2.0) * np.exp((1/v)* ((-c**2/(1-w))-1/(w))) dw
# this is integrated as w=0..1

integrand_new = lambda w,v,c:  v**(-2.0) * w**((-3.0)/2.0) * (1-w)**((-3.0)/2.0) * np.exp((1.0/v)* ((-c**2/(1-w))-1/(w)))  # as w=0..1

dintegranddv=lambda u,v,c: ((-3.0)*u**((-3.0)/2.0)*(v-u)**((-5.0)/2.0)*np.exp((-c**2/(v-u))-1/u))/2.0+c**2*u**((-3.0)/2.0)*(v-u)**((-7.0)/2.0)*np.exp((-c**2/(v-u))-1/u)

dintegranddc=lambda u,v,c: -2*c*u**((-3.0)/2.0)*(v-u)**((-5.0)/2.0)*np.exp((-c**2/(v-u))-1/u)


# dnewintegranddv = lambda w,v,c: -2.0*v**(-3.0) * w**((-3.0)/2.0) * (1-w)**((-3.0)/2.0) * np.exp((1.0/v)* ((-c**2/(1-w))-1/(w)))   +    v**(-2.0) * w**((-3.0)/2.0) * (1-w)**((-3.0)/2.0) * (-np.exp((1/v)* ((-c**2/(1-w))-1/(w)))) * ((-c**2/(1-w))-1/(w)) * v**(-2.0)
dnewintegranddv = lambda w,v,c: -2.0*v**(-3.0) * w**((-3.0)/2.0) * (1-w)**((-3.0)/2.0) * np.exp((1.0/v)* ((-c**2/(1-w))-1/(w)))   -    v**(-4.0) * w**((-3.0)/2.0) * (1-w)**((-3.0)/2.0) * (np.exp((1.0/v)* ((-c**2/(1-w))-1/(w)))) * ((-c**2/(1-w))-1/(w)) 

dnewintegranddc = lambda w,v,c: v**(-2.0) * w**((-3.0)/2.0) * (1-w)**((-3.0)/2.0) * np.exp((1.0/v)* ((-c**2/(1-w))-1/(w))) * (1.0/v) * (-1.0/(1-w))*2.0*c


# # Evaluate integral at a point:
# val=quad(lambda u: integrand(u,v,c),0,v)

integral=np.vectorize(lambda v,c: quad(lambda u: integrand(u,v,c),0,v))

integral_new=np.vectorize(lambda v,c: quad(lambda w: integrand_new(w,v,c),0,1))

integral_dintegranddv=np.vectorize(lambda v,c: quad(lambda u: dintegranddv(u,v,c),0,v))

integral_dintegranddc=np.vectorize(lambda v,c: quad(lambda u: dintegranddc(u,v,c),0,v))

integral_dintegranddv_new=np.vectorize(lambda v,c: quad(lambda w: dnewintegranddv(w,v,c),0,1))

integral_dintegranddc_new=np.vectorize(lambda v,c: quad(lambda w: dnewintegranddc(w,v,c),0,1))


# ***!!! Possible BUG: We use the derivatives of the integrand to find
# the slope with respect to v and c for interpolation, but
# this approach is probably invalid for v since v is also a bound
# of the integration!!!***

# ***!!! BUG now fixed with "new" above

if __name__=="__main__":
    import matplotlib
    import pylab as pl
    from mpl_toolkits.mplot3d import Axes3D
    
    # When run from the command line, generate /tmp/greensconvolution.nc

    # max value at v=.794,c=(0.1 or less)
    vrange=10**np.arange(-5,4,.01,dtype='f')
    crange=10**np.arange(-1,4,.01,dtype='f')  # Note: Convergence trouble for small values of c... fortunately we don't care because c < 1 means 3d propagation is much shorter than 1D distance

    vrange=vrange.reshape(vrange.shape[0],1)
    crange=crange.reshape(1,crange.shape[0])

    
    print("Compute integral")
    integraleval=integral(vrange,crange)
    integraleval_new=integral_new(vrange,crange)
    
    if np.any( (abs(integraleval[0]) > 1e-4*abs(integraleval[1])) &
           (abs(integraleval[1]) > 1e-6) ):
        raise ValueError("Inaccurate integration")

    if np.any( (abs(integraleval_new[0]) > 1e-4*abs(integraleval_new[1])) &
               (abs(integraleval_new[1]) > 1e-6) ):
        raise ValueError("Inaccurate integration")
        
    

    print("Compute integral of derivative along v")
    integral_dintegranddveval=integral_dintegranddv(vrange,crange)
    
    if np.any( (abs(integral_dintegranddveval[0]) > 1e-4*abs(integral_dintegranddveval[1])) &
               (abs(integral_dintegranddveval[1]) > 1e-6) & (crange > 0.35)):
        raise ValueError("Inaccurate integration of derivative along v")

    
    print("Compute integral of derivative along v")
    integral_dintegranddveval_new=integral_dintegranddv_new(vrange,crange)
    
    if np.any( (abs(integral_dintegranddveval_new[0]) > 1e-4*abs(integral_dintegranddveval_new[1])) &
               (abs(integral_dintegranddveval_new[1]) > 1e-6) & (crange > 0.35)):
        raise ValueError("Inaccurate integration of derivative along v")
    
    print("Compute integral of derivative along c")
    integral_dintegranddceval=integral_dintegranddc(vrange,crange)

    if np.any( (abs(integral_dintegranddceval[0]) > 1e-4*abs(integral_dintegranddceval[1])) &
               (abs(integral_dintegranddceval[1]) > 1e-6) & (crange >= 0.4)):
        raise ValueError("Inaccurate integration of derivative along c")

    print("Compute integral of derivative along c")
    integral_dintegranddceval_new=integral_dintegranddc_new(vrange,crange)

    if np.any( (abs(integral_dintegranddceval_new[0]) > 1e-4*abs(integral_dintegranddceval_new[1])) &
               (abs(integral_dintegranddceval_new[1]) > 1e-6) & (crange >= 0.4)):
        raise ValueError("Inaccurate integration of derivative along c")
    
    # np.where((abs(integral_dintegranddceval[0]) > 1e-4*abs(integral_dintegranddceval[1])) & (abs(integral_dintegranddceval[1]) > 1e-6))

    fig=matplotlib.pyplot.figure(1)
    fig.clf()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(np.log10(vrange),np.log10(crange),integraleval[0],rstride=4,cstride=4)
    pl.xlabel('v')
    pl.ylabel('c')
    pl.title('integral')

    fig=matplotlib.pyplot.figure(2)
    fig.clf()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(np.log10(vrange),np.log10(crange),integraleval_new[0],rstride=4,cstride=4)
    pl.xlabel('v')
    pl.ylabel('c')
    pl.title('integral (new)')

    
    fig=matplotlib.pyplot.figure(3)
    fig.clf()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(np.log10(vrange),np.log10(crange),integral_dintegranddveval[0],rstride=4,cstride=4)
    pl.xlabel('v')
    pl.ylabel('c')
    pl.title('integral of dintegranddv')

    fig=matplotlib.pyplot.figure(4)
    fig.clf()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(np.log10(vrange),np.log10(crange),integral_dintegranddveval_new[0],rstride=4,cstride=4)
    pl.xlabel('v')
    pl.ylabel('c')
    pl.title('integral of dintegranddv (new)')

    
    fig=matplotlib.pyplot.figure(5)
    fig.clf()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(np.log10(vrange),np.log10(crange),integral_dintegranddceval[0],rstride=4,cstride=4)
    pl.xlabel('v')
    pl.ylabel('c')
    pl.title('integral of dintegranddc')

    fig=matplotlib.pyplot.figure(6)
    fig.clf()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(np.log10(vrange),np.log10(crange),integral_dintegranddceval_new[0],rstride=4,cstride=4)
    pl.xlabel('v')
    pl.ylabel('c')
    pl.title('integral of dintegranddc')

    # Evaluate at trial point
    trialv=0.8
    trialc=1.03
    print("v=%f, c=%f (old):" % (trialv,trialc))
    evalfromneighbors(vrange,crange,integraleval[0],integral_dintegranddveval[0],integral_dintegranddceval[0],trialv,trialc,verbose=True)

    evalfromneighbors(vrange,crange,integraleval_new[0],integral_dintegranddveval_new[0],integral_dintegranddceval_new[0],trialv,trialc,verbose=True)

    
    trialv2=15
    trialc2=3
    print("v=%f, c=%f (old):" % (trialv2,trialc2))
    evalfromneighbors(vrange,crange,integraleval[0],integral_dintegranddveval[0],integral_dintegranddceval[0],trialv2,trialc2,verbose=True)

    print("v=%f, c=%f (new):" % (trialv2,trialc2))
    evalfromneighbors(vrange,crange,integraleval_new[0],integral_dintegranddveval_new[0],integral_dintegranddceval_new[0],trialv2,trialc2,verbose=True)

    
    trialv3=1500
    trialc3=300
    print("v=%f, c=%f (old):" % (trialv3,trialc3))
    evalfromneighbors(vrange,crange,integraleval[0],integral_dintegranddveval[0],integral_dintegranddceval[0],trialv3,trialc3,verbose=True)
    
    print("v=%f, c=%f (new):" % (trialv3,trialc3))
    evalfromneighbors(vrange,crange,integraleval_new[0],integral_dintegranddveval_new[0],integral_dintegranddceval_new[0],trialv3,trialc3,verbose=True)
    
    trialv4=.5
    trialc4=.95
    print("v=%f, c=%f (old):" % (trialv4,trialc4))
    evalfromneighbors(vrange,crange,integraleval[0],integral_dintegranddveval[0],integral_dintegranddceval[0],trialv4,trialc4,verbose=True)

    print("v=%f, c=%f (new):" % (trialv4,trialc4))
    evalfromneighbors(vrange,crange,integraleval_new[0],integral_dintegranddveval_new[0],integral_dintegranddceval_new[0],trialv4,trialc4,verbose=True)

    trialv5=.005
    trialc5=1.4
    print("v=%f, c=%f (old):" % (trialv5,trialc5))
    evalfromneighbors(vrange,crange,integraleval[0],integral_dintegranddveval[0],integral_dintegranddceval[0],trialv5,trialc5,verbose=True)

    print("v=%f, c=%f (new):" % (trialv5,trialc5))
    evalfromneighbors(vrange,crange,integraleval[0],integral_dintegranddveval_new[0],integral_dintegranddceval_new[0],trialv5,trialc5,verbose=True)



    # Write to NETCDF file
    # Use "NEW" results even though they don't seem to be substantively different
    
    rootgrp=Dataset("/tmp/greensconvolution.nc","w",format="NETCDF4")
    
    vdim=rootgrp.createDimension("v",vrange.shape[0])
    cdim=rootgrp.createDimension("c",crange.shape[1])
    
    vvals=rootgrp.createVariable("v","f4",("v",))
    vvals[:]=vrange
    cvals=rootgrp.createVariable("c","f4",("c",))
    cvals[:]=crange
    integralevalvals=rootgrp.createVariable("integraleval","f4",("v","c"))
    integralevalvals[::]=integraleval_new[0]
    integral_dintegranddvevalvals=rootgrp.createVariable("integral_dintegranddveval","f4",("v","c"))
    integral_dintegranddvevalvals[::]=integral_dintegranddveval_new[0]
    integral_dintegranddcevalvals=rootgrp.createVariable("integral_dintegranddceval","f4",("v","c"))
    integral_dintegranddcevalvals[::]=integral_dintegranddceval_new[0]

    rootgrp.close()
    

