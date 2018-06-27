import sys
import os
import getpass

from greensconvolution.greensconvolution_fast import greensconvolution_integrate
from greensconvolution.greensconvolution_calc import read_greensconvolution

import numpy as np
numpy=np
import pylab as pl
import copy

import datetime

# Fix missing timedelta.total_seconds in Python < 2.7, based on http://stackoverflow.com/questions/3318348/how-can-i-extend-pythons-datetime-datetime-with-my-own-methods/14214646#14214646
import ctypes as c


#matplotlib.rcParams.update({'font.size': 16})


_get_dict = c.pythonapi._PyObject_GetDictPtr
_get_dict.restype = c.POINTER(c.py_object)
_get_dict.argtypes = [c.py_object]

from datetime import timedelta
try:
    timedelta.total_seconds # new in 2.7
except AttributeError:
    def total_seconds(td):
        return float((td.microseconds +
                      (td.seconds + td.days * 24 * 3600) * 10**6)) / 10**6
    d = _get_dict(timedelta)[0]
    d['total_seconds'] = total_seconds

import heatsim2
import heatsim2.surface_temperature


flash_energy=10e3 # J/m^2

calc_greensfcn=True
calc_greensfcn_accel=False
calc_heatsim2=True
load_comsol=False #True

# Create x,y,z voxel center coords
nz=7
ny=32*2
nx=30*2


measx=3e-3
measy=3e-3
#measx=0.0e-3
#measy=0.0e-3

z_thick=1.4e-3 # m

# Flash heating:


# Temperature after first frame should be:
#flash_energy/(composite_rho*composite_c*dz)


(dz,dy,dx,
 z,y,x,
 zgrid,ygrid,xgrid,
 z_bnd,y_bnd,x_bnd,
 z_bnd_z,z_bnd_y,z_bnd_x,
 y_bnd_z,y_bnd_y,y_bnd_x,
 x_bnd_z,x_bnd_y,x_bnd_x,
 r3d,r2d) = heatsim2.build_grid(0,z_thick,nz,
                                -32.e-3,32.e-3,ny,
                                -30.e-3,30.e-3,nx)



measi=np.argmin(abs(x-measx))
measj=np.argmin(abs(y-measy))


# sys.argv[1] be the depth of the hole
# sys.argv[2] will be the filename to write
goaldepth=float(sys.argv[1])
outfilename=sys.argv[2];

barrier_zbnd_idx=np.argmin(abs(z_bnd-goaldepth))
barrier_min_z=z_bnd[barrier_zbnd_idx]
sys.stderr.write("barrier_min_z=%f m\n" % (barrier_min_z));

#barrier_min_z=0.8e-3; 
#barrier_min_z=z_thick 
barrier_min_x=-3e-3
barrier_max_x=3e-3
barrier_min_y=-2.e-3
barrier_max_y=2.e-3

# define materials:
composite_k=.138 # W/m/deg K
composite_rho=1.57e3 # W/m/deg K
composite_c=730 # J/kg/deg K

barrier_k=0 # W/m/deg K
barrier_rho=1.57e3 # W/m/deg K
barrier_c=730 # J/kg/deg K

# final temperature should be:
# width=x_bnd[-1]-x_bnd[0]
# length=y_bnd[-1]-y_bnd[0]
# barrier_width=barrier_max_x-barrier_min_x
# barrier_length=barrier_max_y-barrier_min_y

# flash_energy/(composite_rho*composite_c*(width*length*z_thick-barrier_width*barrier_length*(z_thick-barrier_min_z))/(length*width))
# Or a reasonable approximation
# (neglecting subtracting the hole from the volume -- which is .26%)
# flash_energy/(composite_rho*composite_c*z_thick)





t0=0.0
tf=20.0

nt_heatsim2=200
(t_heatsim2,dt_heatsim2)=np.linspace(t0,tf,num=nt_heatsim2,retstep=True)


greensconvolution_params=read_greensconvolution()



(ymat,xmat)=np.meshgrid(y,x,indexing='ij')
zmat=np.ones((ny,nx),dtype='d')*z_thick
zmat[ (xmat > barrier_min_x) & (xmat < barrier_max_x) &
      (ymat > barrier_min_y) & (ymat < barrier_max_y)]=barrier_min_z
zvec=np.reshape(zmat,np.prod(zmat.shape))
        

# accelerate green's function by using theory for back face
(ymat,xmat)=np.meshgrid(y,x,indexing='ij')
zmat_accel=np.ones((ny,nx),dtype='d')*z_thick
barrier_location=((xmat > barrier_min_x) & (xmat < barrier_max_x) &
                  (ymat > barrier_min_y) & (ymat < barrier_max_y))
zmat_accel[barrier_location]=barrier_min_z
zvec_accel=np.reshape(zmat_accel[barrier_location],np.count_nonzero(barrier_location))

zmat_backing_accel=np.ones((ny,nx),dtype='d')*z_thick
zvec_backing_accel=np.reshape(zmat_backing_accel[barrier_location],np.count_nonzero(barrier_location))


Tg_accel=np.zeros((nt_heatsim2+1,ny,nx),dtype='d')
Tg_accel[1:,::]=((flash_energy/(composite_rho*composite_c))/np.sqrt(np.pi*(composite_k/(composite_rho*composite_c))*(t_heatsim2+dt_heatsim2/2.0))*(1+2*np.exp(-(2*z_thick)**2/(4*(composite_k/(composite_rho*composite_c))*(t_heatsim2+dt_heatsim2/2.0)))+2*np.exp(-(4*z_thick)**2/(4*(composite_k/(composite_rho*composite_c))*(t_heatsim2+dt_heatsim2/2.0))))).reshape((nt_heatsim2,1,1))
    
for jidx in range(ny):
    print("j=%d/%d" % (jidx,ny))
    for iidx in range(nx):
        # print("i=%d/%d" % (iidx,nx))
        # if iidx!=measi or jidx!=measj:
        #     continue
        
        dxmat=x[iidx]-xmat
        dymat=y[jidx]-ymat
        rmat=np.sqrt(dxmat**2.0+dymat**2.0+zmat**2.0)
        rvec=np.reshape(rmat[barrier_location],np.count_nonzero(barrier_location))

        rmat_extraterm=np.sqrt(dxmat**2.0+dymat**2.0+(zmat*3)**2.0) # for extra image sources of barrier
        rvec_extraterm=np.reshape(rmat_extraterm[barrier_location],np.count_nonzero(barrier_location))
        
        
        rmat_backing=np.sqrt(dxmat**2.0+dymat**2.0+zmat_backing_accel**2.0)
        rvec_backing=np.reshape(rmat_backing[barrier_location],np.count_nonzero(barrier_location))
            
        # WARNING: 2.0 in leading coefficient here is a fudge factor!... where did we drop it???
        # Answer: We didn't. There's an image source reflected in the flash
        # plane required to satisfy the no-flow boundary condition on the
        # flash plane. 
        Tg_accel[1:,jidx,iidx]+=dx*dy*flash_energy*2.0*(
            greensconvolution_integrate(greensconvolution_params,zvec_accel,rvec,t_heatsim2+dt_heatsim2/2.0,composite_k,composite_rho,composite_c)-
            greensconvolution_integrate(greensconvolution_params,zvec_backing_accel,rvec_backing,t_heatsim2+dt_heatsim2/2.0,composite_k,composite_rho,composite_c)+
            greensconvolution_integrate(greensconvolution_params,zvec_accel,rvec_extraterm,t_heatsim2+dt_heatsim2/2.0,composite_k,composite_rho,composite_c))
        
        pass
    pass

                                    
np.savetxt(outfilename,np.array((t_heatsim2+dt_heatsim2/2.0,Tg_accel[1:,measj,measi]),dtype='d').T);

