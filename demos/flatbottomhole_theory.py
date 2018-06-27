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
calc_greensfcn_accel=True
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


t0=0.0
tf=20.0

nt_heatsim2=200
(t_heatsim2,dt_heatsim2)=np.linspace(t0,tf,num=nt_heatsim2,retstep=True)



if (measx >= barrier_min_x and measx <= barrier_max_x and
    measy >= barrier_min_y and measy <= barrier_max_y):

    # with flaw
    np.savetxt(outfilename,np.array((t_heatsim2[1:],((flash_energy/(composite_rho*composite_c))/np.sqrt(np.pi*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))*(1+2*np.exp(-(2*barrier_min_z)**2/(4*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))+2*np.exp(-(4*barrier_min_z)**2/(4*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))+2*np.exp(-(6*barrier_min_z)**2/(4*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:])))),dtype='d').T)


    pass
else: 
    # no flaw
    np.savetxt(outfilename,np.array((t_heatsim2[1:],((flash_energy/(composite_rho*composite_c))/np.sqrt(np.pi*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))*(1+2*np.exp(-(2*z_thick)**2/(4*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))+2*np.exp(-(4*z_thick)**2/(4*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:])))),dtype='d').T)
    pass

