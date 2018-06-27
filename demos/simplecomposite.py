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


flash_energy=10e3 # J/m^2

calc_heatsim2=True

# Create x,y,z voxel center coords
nz=7
ny=32 
nx=30 


#measx=0.4e-3
#measy=0.3e-3
measx=0.0e-3
measy=0.0e-3

z_thick=1.4e-3 # m

# Flash heating:


# Temperature after first frame should be:
#flash_energy/(composite_rho*composite_c*dz)

# final temperature should be flash_energy/(composite_rho*composite_c*z_thick)


(dz,dy,dx,
 z,y,x,
 zgrid,ygrid,xgrid,
 z_bnd,y_bnd,x_bnd,
 z_bnd_z,z_bnd_y,z_bnd_x,
 y_bnd_z,y_bnd_y,y_bnd_x,
 x_bnd_z,x_bnd_y,x_bnd_x,
 r3d,r2d) = heatsim2.build_grid(0,z_thick,nz,
                                -16.e-3,16.e-3,ny,
                                -15.e-3,15.e-3,nx)

measi=np.argmin(abs(x-measx))
measj=np.argmin(abs(y-measy))




# define materials:
composite_k=.138 # W/m/deg K
composite_rho=1.57e3 # W/m/deg K
composite_c=730 # J/kg/deg K


materials=(
    # material 0: composite
    (heatsim2.TEMPERATURE_COMPUTE,composite_k,composite_rho,composite_c),
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
# (none)

volumetric_elements[0,:,:]=1  # set flash source (for heatsim2)



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


greensconvolution_params=read_greensconvolution('greensconvolution.nc')



(ymat,xmat)=np.meshgrid(y,x,indexing='ij')
zmat=np.ones((ny,nx),dtype='d')*z_thick
zvec=np.reshape(zmat,np.prod(zmat.shape))
        
Tg=np.zeros((nt_heatsim2+1,ny,nx),dtype='d')
Tg[1:,::]=((flash_energy/(composite_rho*composite_c))/sqrt(np.pi*(composite_k/(composite_rho*composite_c))*(t_heatsim2+dt_heatsim2/2.0))).reshape((nt_heatsim2,1,1))

for jidx in range(ny):
    print("j=%d/%d" % (jidx,ny))
    for iidx in range(nx):
        print("i=%d/%d" % (iidx,nx))
        if iidx!=measi or jidx!=measj:
            continue
        
        dxmat=x[iidx]-xmat
        dymat=y[jidx]-ymat
        rmat=np.sqrt(dxmat**2.0+dymat**2.0+zmat**2.0)
        rvec=np.reshape(rmat,np.prod(rmat.shape))

        # WARNING: 2.0 in leading coefficient here is a fudge factor!... where did we drop it???
        # Answer: We didn't. There's an image source reflected in the flash
        # plane required to satisfy the no-flow boundary condition on the
        # flash plane. 
        Tg[1:,jidx,iidx]+=dx*dy*flash_energy*2.0*greensconvolution_integrate(greensconvolution_params,zvec,rvec,t_heatsim2+dt_heatsim2/2.0,composite_k,composite_rho,composite_c)
        pass
    pass

                                    
if calc_heatsim2:
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
        heatsim2meas[tcnt]=T[tcnt+1,0,measj,measi] 
    
        if tcnt==np.argmin(abs(t_heatsim2-2.0)):
            heatsim2_twosec=copy.copy(T)
            pass
        pass
    heatsim2res=T
    hs2_setuptime=(hs2_start_exec-hs2_start).total_seconds()
    hs2_comptime=(datetime.datetime.now()-hs2_start_exec).total_seconds()
    pass


plotargs=[t_heatsim2[1:],((flash_energy/(composite_rho*composite_c))/sqrt(np.pi*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))*(1+2*exp(-(2*z_thick)**2/(4*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))+2*exp(-(4*z_thick)**2/(4*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))),'--']
legendargs=['Theory']

if calc_heatsim2:
    plotargs.extend([t_heatsim2[1:],heatsim2meas[1:],'x:'])
    legendargs.append('heatsim2 %f setup %f exec' % (hs2_setuptime,hs2_comptime))
    pass

plotargs.extend([t_heatsim2+dt_heatsim2/2.0,Tg[1:,measj,measi],'o-'])
legendargs.append('greensconvolution')

pl.loglog(*plotargs)
pl.legend(legendargs)
pl.xlabel('Time (s)')
pl.ylabel('Temperature (K)')
pl.grid()
pl.show()
