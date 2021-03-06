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



materials=(
    # material 0: composite
    (heatsim2.TEMPERATURE_COMPUTE,composite_k,composite_rho,composite_c),
    # material 1: barrier
    (heatsim2.TEMPERATURE_COMPUTE,barrier_k,barrier_rho,barrier_c),
    # material 2: composite  (so we can use same material matrix as heatsim)
    (heatsim2.TEMPERATURE_COMPUTE,composite_k,composite_rho,composite_c),
    # material 3: Fixed temperature (Dirichlet boundary condition)
    (heatsim2.TEMPERATURE_FIXED,),
)

boundaries=(
    # boundary 0: conducting
    (heatsim2.boundary_conducting,),
    # boundary 1: insulating
    (heatsim2.boundary_insulating,),
)

volumetric=(  # on material grid
    # 0: nothing
    (heatsim2.NO_SOURCE,),
    #1: impulse source @ t=0
    (heatsim2.IMPULSE_SOURCE,0.0,flash_energy/dz), # t (sec), Energy J/m^2
)


# initialize all elements to zero
(material_elements,
 boundary_z_elements,
 boundary_y_elements,
 boundary_x_elements,
 volumetric_elements)=heatsim2.zero_elements(nz,ny,nx) 


# define nonzero material elements

material_elements[(zgrid >= barrier_min_z) &
                  (ygrid >= barrier_min_y) &
                  (ygrid <= barrier_max_y) &
                  (xgrid >= barrier_min_x) &
                  (xgrid <= barrier_max_x)]=1 # material 1: barrier

volumetric_elements[0,:,:]=1  # set flash source (for heatsim2)


# set boundaries of barrier to insulating
boundary_z_elements[ (z_bnd_x > barrier_min_x) &
                     (z_bnd_x < barrier_max_x) &
                     (z_bnd_y > barrier_min_y) &
                     (z_bnd_y < barrier_max_y) &
                     (z_bnd_z==z_bnd[(np.abs(z_bnd-barrier_min_z)).argmin()])]=1 #

boundary_x_elements[ ((x_bnd_x == x_bnd[np.abs(x_bnd-barrier_min_x).argmin()])|
                      (x_bnd_x == x_bnd[np.abs(x_bnd-barrier_max_x).argmin()])) &
                     (x_bnd_y > barrier_min_y) &
                     (x_bnd_y < barrier_max_y) &
                     (x_bnd_z > barrier_min_z)]=1

boundary_y_elements[ ((y_bnd_y == y_bnd[np.abs(y_bnd-barrier_min_y).argmin()])|
                      (y_bnd_y == y_bnd[np.abs(y_bnd-barrier_max_y).argmin()]))&
                     (y_bnd_x > barrier_min_x) &
                     (y_bnd_x < barrier_max_x) &
                     (y_bnd_z > barrier_min_z)]=1

# set edges to insulating
boundary_x_elements[:,:,0]=1 # insulating
boundary_x_elements[:,:,-1]=1 # insulating
boundary_y_elements[:,0,:]=1 # insulating
boundary_y_elements[:,-1,:]=1 # insulating
boundary_z_elements[0,:,:]=1 # insulating
boundary_z_elements[-1,:,:]=1 # insulating


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
        
                                    
hs2_start=datetime.datetime.now()
(ADI_params,ADI_steps)=heatsim2.setup(z[0],y[0],x[0],
                                      dz,dy,dx,
                                      nz,ny,nx,
                                      dt_heatsim2,
                                      materials,
                                      boundaries,
                                      volumetric,
                                      material_elements,
                                      boundary_z_elements,
                                      boundary_y_elements,
                                      boundary_x_elements,
                                      volumetric_elements)


T=np.zeros((nt_heatsim2+1,nz,ny,nx),dtype='d')

hs2_start_exec=datetime.datetime.now()
heatsim2meas=np.zeros(nt_heatsim2,dtype='d')
for tcnt in range(nt_heatsim2):
    curt=t0+dt_heatsim2*tcnt
    print "t=%f" % (curt)
    T[tcnt+1,::]=heatsim2.run_adi_steps(ADI_params,ADI_steps,curt,dt_heatsim2,T[tcnt,::],volumetric_elements,volumetric)
    heatsim2meas[tcnt]=heatsim2.surface_temperature.insulating_z_min_surface_temperature(T[tcnt+1,0:2,measj:measj+1,measi:measi+1],dz)
    # heatsim2meas[tcnt]=T[tcnt+1,0,measj,measi]
    
    if tcnt==np.argmin(abs(t_heatsim2-2.0)):
        heatsim2_twosec=copy.copy(T)
        pass
    pass
heatsim2res=T
hs2_setuptime=(hs2_start_exec-hs2_start).total_seconds()
hs2_comptime=(datetime.datetime.now()-hs2_start_exec).total_seconds()

np.savetxt(outfilename,np.array((t_heatsim2[1:],heatsim2meas[1:]),dtype='d').T)
