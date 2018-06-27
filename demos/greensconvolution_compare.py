import sys
import os
import getpass

from greensconvolution.greensconvolution_fast import greensconvolution_integrate
from greensconvolution.greensconvolution_calc import read_greensconvolution

import matplotlib
import numpy as np
numpy=np
import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

import datetime

# Fix missing timedelta.total_seconds in Python < 2.7, based on http://stackoverflow.com/questions/3318348/how-can-i-extend-pythons-datetime-datetime-with-my-own-methods/14214646#14214646
import ctypes as c


matplotlib.rcParams.update({'font.size': 16})


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
calc_greensfcn_dumbaccel=False
calc_greensfcn_simple=True
calc_heatsim2=True
load_comsol=False #True

#gc_kernel="opencl_interpolator" # greensconvolution kernel to use .. laptop: 18 sec
#gc_kernel="openmp_interpolator" # laptop: 19 sec
gc_kernel="opencl_quadpack"   # laptop: 1 min 10 sec
#gc_kernel="opencl_simplegaussquad"   # Laptop: VERY slow

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



barrier_min_z=0.8e-3; 
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
zvec=zmat.reshape(np.prod(zmat.shape),1)

if calc_greensfcn:
    Tg=np.zeros((nt_heatsim2+1,ny,nx),dtype='d')
    Tg[1:,::]=((flash_energy/(composite_rho*composite_c))/np.sqrt(np.pi*(composite_k/(composite_rho*composite_c))*(t_heatsim2+dt_heatsim2/2.0))).reshape((nt_heatsim2,1,1))

    for jidx in range(ny):
        print("j=%d/%d" % (jidx,ny))
        for iidx in range(nx):
            # print("i=%d/%d" % (iidx,nx))
            if iidx!=measi or jidx!=measj:
                continue
        
            dxmat=x[iidx]-xmat
            dymat=y[jidx]-ymat
            rmat=np.sqrt(dxmat**2.0+dymat**2.0+zmat**2.0)
            rvec=rmat.reshape(np.prod(rmat.shape),1)

            rmat_extraterm=np.sqrt(dxmat**2.0+dymat**2.0+(zmat*3)**2.0) # for extra image sources of barrier
            rvec_extraterm=rmat_extraterm.reshape(np.prod(rmat_extraterm.shape),1)

            # WARNING: 2.0 in leading coefficient here is a fudge factor!... where did we drop it???
            # Answer: We didn't. There's an image source reflected in the flash
            # plane required to satisfy the no-flow boundary condition on the
            # flash plane. 
            Tg[1:,jidx,iidx]+=(greensconvolution_integrate(greensconvolution_params,zvec.astype(np.float32),rvec.astype(np.float32),t_heatsim2.astype(np.float32).reshape(1,nt_heatsim2)+dt_heatsim2/2.0,composite_k,composite_rho,composite_c,dx*dy*flash_energy*2.0,(0,),kernel=gc_kernel)+greensconvolution_integrate(greensconvolution_params,zvec.astype(np.float32),rvec_extraterm.astype(np.float32),t_heatsim2.astype(np.float32).reshape(1,nt_heatsim2)+dt_heatsim2/2.0,composite_k,composite_rho,composite_c,dx*dy*flash_energy*2.0,(0,),kernel=gc_kernel))
            pass
        pass
    pass

if calc_greensfcn_accel:
    # accelerate green's function by using theory for back face
    (ymat,xmat)=np.meshgrid(y,x,indexing='ij')
    zmat_accel=np.ones((ny,nx),dtype='d')*z_thick
    barrier_location=((xmat > barrier_min_x) & (xmat < barrier_max_x) &
                      (ymat > barrier_min_y) & (ymat < barrier_max_y))
    zmat_accel[barrier_location]=barrier_min_z
    zvec_accel=zmat_accel[barrier_location].reshape(np.count_nonzero(barrier_location),1)
    
    zmat_backing_accel=np.ones((ny,nx),dtype='d')*z_thick
    zvec_backing_accel=zmat_backing_accel[barrier_location].reshape(np.count_nonzero(barrier_location),1)
    
    
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
            rvec=rmat[barrier_location].reshape(np.count_nonzero(barrier_location),1)
            
            rmat_extraterm=np.sqrt(dxmat**2.0+dymat**2.0+(zmat*3)**2.0) # for extra image sources of barrier
            rvec_extraterm=rmat_extraterm[barrier_location].reshape(np.count_nonzero(barrier_location),1)
            
            
            rmat_backing=np.sqrt(dxmat**2.0+dymat**2.0+zmat_backing_accel**2.0)
            rvec_backing=rmat_backing[barrier_location].reshape(np.count_nonzero(barrier_location),1)
            
            # WARNING: 2.0 in leading coefficient here is a fudge factor!... where did we drop it???
            # Answer: We didn't. There's an image source reflected in the flash
            # plane required to satisfy the no-flow boundary condition on the
            # flash plane. 
            Tg_accel[1:,jidx,iidx]+=dx*dy*flash_energy*2.0*(
                greensconvolution_integrate(greensconvolution_params,zvec_accel.astype(np.float32),rvec.astype(np.float32),t_heatsim2.astype(np.float32).reshape(1,nt_heatsim2)+dt_heatsim2/2.0,composite_k,composite_rho,composite_c,1.0,(0,),kernel=gc_kernel)-
                greensconvolution_integrate(greensconvolution_params,zvec_backing_accel.astype(np.float32),rvec_backing.astype(np.float32),t_heatsim2.astype(np.float32).reshape(1,nt_heatsim2)+dt_heatsim2/2.0,composite_k,composite_rho,composite_c,1.0,(0,),kernel=gc_kernel)+
                greensconvolution_integrate(greensconvolution_params,zvec_accel.astype(np.float32),rvec_extraterm.astype(np.float32),t_heatsim2.astype(np.float32).reshape(1,nt_heatsim2)+dt_heatsim2/2.0,composite_k,composite_rho,composite_c,1.0,(0,),kernel=gc_kernel))
            
            pass
        pass
    pass


if calc_greensfcn_simple:
    # just instantiate Green's function at twice depth
    (ymat,xmat)=np.meshgrid(y,x,indexing='ij')
    zmat_accel=np.ones((ny,nx),dtype='d')*z_thick
    barrier_location=((xmat > barrier_min_x) & (xmat < barrier_max_x) &
                      (ymat > barrier_min_y) & (ymat < barrier_max_y))
    zmat_accel[barrier_location]=barrier_min_z
    zvec_accel=np.reshape(zmat_accel[barrier_location],np.count_nonzero(barrier_location))
    
    zmat_backing_accel=np.ones((ny,nx),dtype='d')*z_thick
    zvec_backing_accel=np.reshape(zmat_backing_accel[barrier_location],np.count_nonzero(barrier_location))

    
    Tg_simpleaccel=np.zeros((nt_heatsim2+1,ny,nx),dtype='d')
    Tg_simpleaccel[1:,::]=((flash_energy/(composite_rho*composite_c))/np.sqrt(np.pi*(composite_k/(composite_rho*composite_c))*(t_heatsim2+dt_heatsim2/2.0))*(1+2*np.exp(-(2*z_thick)**2/(4*(composite_k/(composite_rho*composite_c))*(t_heatsim2+dt_heatsim2/2.0)))+2*np.exp(-(4*z_thick)**2/(4*(composite_k/(composite_rho*composite_c))*(t_heatsim2+dt_heatsim2/2.0))))).reshape((nt_heatsim2,1,1))

    num_rs=np.count_nonzero(barrier_location)
    
    for jidx in range(ny):
        print("j=%d/%d" % (jidx,ny))
        for iidx in range(nx):
            # print("i=%d/%d" % (iidx,nx))
            # if iidx!=measi or jidx!=measj:
            #     continue

            
            dxmat=x[iidx]-xmat
            dymat=y[jidx]-ymat
            rmat=np.sqrt(dxmat**2.0+dymat**2.0+(2.0*zmat)**2.0)
            rvec=np.reshape(rmat[barrier_location],num_rs)

            rmat_extraterm=np.sqrt(dxmat**2.0+dymat**2.0+(zmat*4)**2.0) # for extra image sources of barrier
            rvec_extraterm=np.reshape(rmat_extraterm[barrier_location],num_rs)
            
            rmat_extraterm2=np.sqrt(dxmat**2.0+dymat**2.0+(zmat*6)**2.0) # for extra image sources of barrier
            rvec_extraterm2=np.reshape(rmat_extraterm2[barrier_location],num_rs)
            
            
            rmat_backing=np.sqrt(dxmat**2.0+dymat**2.0+(zmat_backing_accel*2.0)**2.0)
            rvec_backing=np.reshape(rmat_backing[barrier_location],num_rs)

            rmat_backing_extraterm=np.sqrt(dxmat**2.0+dymat**2.0+(zmat_backing_accel*4.0)**2.0)
            rvec_backing_extraterm=np.reshape(rmat_backing_extraterm[barrier_location],num_rs)

            # WARNING: 2.0 in leading coefficient here is a fudge factor!... where did we drop it???
            # Answer: We didn't. There's an image source reflected in the flash
            # plane required to satisfy the no-flow boundary condition on the
            # flash plane. 
            Tg_simpleaccel[1:,jidx,iidx]+=dx*dy*(2.0*((2.0*flash_energy/(composite_rho*composite_c))/(4.0*np.pi*(composite_k/(composite_rho*composite_c))*(t_heatsim2.reshape(nt_heatsim2,1)+dt_heatsim2/2.0))**(3.0/2.0))*(np.exp(-rvec.reshape(1,num_rs)**2.0/(4.0*(composite_k/(composite_rho*composite_c))*(t_heatsim2.reshape(nt_heatsim2,1)+dt_heatsim2/2.0))) - np.exp(-rvec_backing.reshape(1,num_rs)**2.0/(4.0*(composite_k/(composite_rho*composite_c))*(t_heatsim2.reshape(nt_heatsim2,1)+dt_heatsim2/2.0))) + np.exp(-rvec_extraterm.reshape(1,num_rs)**2.0/(4.0*(composite_k/(composite_rho*composite_c))*(t_heatsim2.reshape(nt_heatsim2,1)+dt_heatsim2/2.0))) + np.exp(-rvec_extraterm2.reshape(1,num_rs)**2.0/(4.0*(composite_k/(composite_rho*composite_c))*(t_heatsim2.reshape(nt_heatsim2,1)+dt_heatsim2/2.0))) - np.exp(-rvec_backing_extraterm.reshape(1,num_rs)**2.0/(4.0*(composite_k/(composite_rho*composite_c))*(t_heatsim2.reshape(nt_heatsim2,1)+dt_heatsim2/2.0))) )).sum(1)


            
            pass
        pass
    pass
    


if calc_greensfcn_dumbaccel:
    # accelerate green's function but neglect change to back face
    (ymat,xmat)=np.meshgrid(y,x,indexing='ij')
    zmat_accel=np.ones((ny,nx),dtype='d')*z_thick
    barrier_location=((xmat > barrier_min_x) & (xmat < barrier_max_x) &
                      (ymat > barrier_min_y) & (ymat < barrier_max_y))
    zmat_accel[barrier_location]=barrier_min_z
    zvec_accel=zmat_accel[barrier_location].reshape(np.count_nonzero(barrier_location),1)
    
    
    Tg_dumbaccel=np.zeros((nt_heatsim2+1,ny,nx),dtype='d')
    Tg_dumbaccel[1:,::]=((flash_energy/(composite_rho*composite_c))/np.sqrt(np.pi*(composite_k/(composite_rho*composite_c))*(t_heatsim2+dt_heatsim2/2.0))*(1+2*np.exp(-(2*z_thick)**2/(4*(composite_k/(composite_rho*composite_c))*(t_heatsim2+dt_heatsim2/2.0)))+2*np.exp(-(4*z_thick)**2/(4*(composite_k/(composite_rho*composite_c))*(t_heatsim2+dt_heatsim2/2.0))))).reshape((nt_heatsim2,1,1))
    
    for jidx in range(ny):
        print("j=%d/%d" % (jidx,ny))
        for iidx in range(nx):
            # print("i=%d/%d" % (iidx,nx))
            # if iidx!=measi or jidx!=measj:
            #     continue
        
            dxmat=x[iidx]-xmat
            dymat=y[jidx]-ymat
            rmat=np.sqrt(dxmat**2.0+dymat**2.0+zmat**2.0)
            rvec=rmat[barrier_location].reshape(np.count_nonzero(barrier_location),1)
            
            rmat_extraterm=np.sqrt(dxmat**2.0+dymat**2.0+(zmat*3)**2.0) # for extra image sources of barrier
            rvec_extraterm=rmat_extraterm[barrier_location].reshape(np.count_nonzero(barrier_location),1)
            
            
            # WARNING: 2.0 in leading coefficient here is a fudge factor!... where did we drop it???
            # Answer: We didn't. There's an image source reflected in the flash
            # plane required to satisfy the no-flow boundary condition on the
            # flash plane. 
            Tg_dumbaccel[1:,jidx,iidx]+=dx*dy*flash_energy*2.0*(
                greensconvolution_integrate(greensconvolution_params,zvec_accel.astype(np.float32),rvec.astype(np.float32),t_heatsim2.astype(np.float32).reshape(1,nt_heatsim2)+dt_heatsim2/2.0,composite_k,composite_rho,composite_c,1.0,(0,),kernel=gc_kernel)+
                greensconvolution_integrate(greensconvolution_params,zvec_accel.astype(np.float32),rvec_extraterm.astype(np.float32),t_heatsim2.astype(np.float32).reshape(1,nt_heatsim2)+dt_heatsim2/2.0,composite_k,composite_rho,composite_c,1.0,(0,),kernel=gc_kernel))
            
            pass
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
        heatsim2meas[tcnt]=heatsim2.surface_temperature.insulating_z_min_surface_temperature(T[tcnt+1,0:2,measj:measj+1,measi:measi+1],dz)
        #heatsim2meas[tcnt]=T[tcnt+1,0,measj,measi]
    
        if tcnt==np.argmin(abs(t_heatsim2-2.0)):
            heatsim2_twosec=copy.copy(T)
            pass
        pass
    heatsim2res=T
    hs2_setuptime=(hs2_start_exec-hs2_start).total_seconds()
    hs2_comptime=(datetime.datetime.now()-hs2_start_exec).total_seconds()
    pass

if load_comsol:
    comsoloutput=os.path.join('/tmp','heatsimcomsoloutput_%s.csv' % (getpass.getuser()))
    comsoldat=np.genfromtxt(comsoloutput, delimiter=',',comments='%')
    t_comsol=comsoldat[:,0][comsoldat[:,0] > 0.0]
    T_comsol=comsoldat[:,1][comsoldat[:,0] > 0.0]#-293.15;
    pass



plotargs=[t_heatsim2[1:],((flash_energy/(composite_rho*composite_c))/np.sqrt(np.pi*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))*(1+2*np.exp(-(2*z_thick)**2/(4*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))+2*np.exp(-(4*z_thick)**2/(4*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))),'--']
legendargs=['1D Theory (no flaw)']

plotargs.extend([t_heatsim2[1:],((flash_energy/(composite_rho*composite_c))/np.sqrt(np.pi*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))*(1+2*np.exp(-(2*barrier_min_z)**2/(4*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))+2*np.exp(-(4*barrier_min_z)**2/(4*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))+2*np.exp(-(6*barrier_min_z)**2/(4*(composite_k/(composite_rho*composite_c))*t_heatsim2[1:]))),'--'])
legendargs.append('1D Theory (with flaw)')

if calc_heatsim2:
    plotargs.extend([t_heatsim2[1:],heatsim2meas[1:],'x:'])
    #legendargs.append('heatsim2 %f setup %f exec' % (hs2_setuptime,hs2_comptime))
    legendargs.append('Finite difference')
    pass

if calc_greensfcn:
    plotargs.extend([t_heatsim2+dt_heatsim2/2.0,Tg[1:,measj,measi],'o-'])
    legendargs.append('Integral of Green\'s functions')
    pass

if calc_greensfcn_accel:
    plotargs.extend([t_heatsim2+dt_heatsim2/2.0,Tg_accel[1:,measj,measi],'o-'])
    legendargs.append('Integral of Green\'s functions (accelerated)')
    pass

if calc_greensfcn_simple:
    plotargs.extend([t_heatsim2+dt_heatsim2/2.0,Tg_simpleaccel[1:,measj,measi],'o-'])
    legendargs.append('Simpler integral of Green\'s functions (accelerated)')
    pass

if load_comsol:
    plotargs.extend([t_comsol,T_comsol,'*-.'])
    legendargs.append('COMSOL')
    pass


pl.figure(1)
pl.clf()
pl.loglog(*plotargs,markersize=8,linewidth=4,markeredgewidth=1)
pl.legend(legendargs,fontsize=12)
pl.xlabel('Time (s)')
pl.ylabel('Temperature (K)')
pl.grid()
pl.axis([1e-2,20,5,100])
pl.savefig("/tmp/flatbottomhole.png",dpi=300)
pl.show()


pl.figure(2)
pl.clf()
pl.loglog(*plotargs,markersize=8,linewidth=4,markeredgewidth=1)
pl.legend(legendargs,fontsize=12)
pl.xlabel('Time (s)')
pl.ylabel('Temperature (K)')
pl.grid()
pl.axis([1,20,6,15])
pl.savefig("/tmp/flatbottomhole_zoom.png",dpi=300)
pl.show()

# build log10(t) vs x images
logtrange=np.arange(-0.8,1.2,0.1,dtype='d')
ypos=0.0005
yidx=np.argmin(abs(y-.0005))

fdimage=np.zeros((logtrange.shape[0],x.shape[0]),dtype='d')
gfimage=np.zeros((logtrange.shape[0],x.shape[0]),dtype='d')
sgfimage=np.zeros((logtrange.shape[0],x.shape[0]),dtype='d')
#dgfimage=np.zeros((logtrange.shape[0],x.shape[0]),dtype='d')
for logtcnt in range(logtrange.shape[0]):
    logtval=logtrange[logtcnt]
    tval=10**logtval
    
    nextidx=np.where(t_heatsim2 >= tval)[0][0]
    previdx=nextidx-1
    
    nextlogt=np.log10(t_heatsim2[nextidx])
    prevlogt=np.log10(t_heatsim2[previdx])
    
    
    nexths2values=np.log10(heatsim2.surface_temperature.insulating_z_min_surface_temperature(T[nextidx,0:2,yidx,:],dz))
    prevhs2values=np.log10(heatsim2.surface_temperature.insulating_z_min_surface_temperature(T[previdx,0:2,yidx,:],dz))
    
    interphs2values=prevhs2values + (nexths2values-prevhs2values)*((logtval-prevlogt)/(nextlogt-prevlogt))
    
    nextgfvalues=np.log10(Tg_accel[nextidx,yidx,:])
    prevgfvalues=np.log10(Tg_accel[previdx,yidx,:])
    
    interpgfvalues=prevgfvalues + (nextgfvalues-prevgfvalues)*((logtval-prevlogt)/(nextlogt-prevlogt))
    
    #nextdgfvalues=np.log10(Tg_dumbaccel[nextidx,yidx,:])
    #prevdgfvalues=np.log10(Tg_dumbaccel[previdx,yidx,:])
    
    #interpdgfvalues=prevdgfvalues + (nextdgfvalues-prevdgfvalues)*((logtval-prevlogt)/(nextlogt-prevlogt))

    nextsgfvalues=np.log10(Tg_simpleaccel[nextidx,yidx,:])
    prevsgfvalues=np.log10(Tg_simpleaccel[previdx,yidx,:])
    
    interpsgfvalues=prevsgfvalues + (nextsgfvalues-prevsgfvalues)*((logtval-prevlogt)/(nextlogt-prevlogt))

    
    fdimage[logtcnt,:]=interphs2values
    gfimage[logtcnt,:]=interpgfvalues
    sgfimage[logtcnt,:]=interpsgfvalues
    #dgfimage[logtcnt,:]=interpdgfvalues
    pass

f3=pl.figure(3)
pl.clf()
f3im=pl.imshow(fdimage,vmin=0.8,vmax=1.65)
pl.title('Finite difference (log10)')
pl.xlabel('X position')
pl.ylabel('log10(time)')
#f3im.get_axes().set_aspect(1.8)
divider3=make_axes_locatable(f3im.get_axes())
cax3 = divider3.append_axes("right", size="5%", pad=0.05)
f3cb=pl.colorbar(f3im,cax=cax3)

pl.savefig('/tmp/greensconvolution_compare_finitediff.png',bbox_inches='tight')


f4=pl.figure(4)
pl.clf()
f4im=pl.imshow(gfimage,vmin=0.8,vmax=1.65)
#f4ax.get_axes().set_aspect(1.8)
pl.title('Accelerated Green\'s function (log10)')
pl.xlabel('X position')
pl.ylabel('log10(time)')
divider4=make_axes_locatable(f4im.get_axes())
cax4 = divider4.append_axes("right", size="5%", pad=0.05)
f4cb=pl.colorbar(f4im,cax=cax4)
pl.savefig('/tmp/greensconvolution_compare_accelgreenfn.png',bbox_inches='tight')
#f4cb.ax.set_aspect(1.9)

pl.figure(5)
pl.clf()
f5im=pl.imshow(sgfimage,vmin=0.8,vmax=1.65)
pl.title('Accelerated Simple Green\'s function (log10)')
pl.xlabel('X position')
pl.ylabel('log10(time)')
divider5=make_axes_locatable(f5im.get_axes())
cax5 = divider5.append_axes("right", size="5%", pad=0.05)
f5cb=pl.colorbar(f5im,cax=cax5)
pl.savefig('/tmp/greensconvolution_compare_accelsimple.png',bbox_inches='tight')


xsplit=0
xsplitidx=np.argmin(abs(x_bnd-xsplit))

splitimg=np.concatenate((fdimage[:,:xsplit],gfimage[:,xsplit:]),axis=1)

splitimg2=np.concatenate((fdimage[:,:xsplit],sgfimage[:,xsplit:]),axis=1)

pl.figure(6)
pl.clf()
f6im=pl.imshow(splitimg,vmin=0.8,vmax=1.65)
pl.title('L: Finite difference; R: Accel. Green\'s function (log10)')
pl.xlabel('X position')
pl.ylabel('log10(time)')
divider6=make_axes_locatable(f6im.get_axes())
cax6 = divider6.append_axes("right", size="5%", pad=0.05)
f6cb=pl.colorbar(f6im,cax=cax6)
pl.savefig('/tmp/greensconvolution_compare_lr_accel.png',bbox_inches='tight')

pl.figure(7)
pl.clf()
f7im=pl.imshow(fdimage-gfimage,vmin=-.04,vmax=.04)
pl.title('log10(finite difference)-log10(Accel. Green\'s function)')
pl.xlabel('X position')
pl.ylabel('log10(time)')
divider7=make_axes_locatable(f7im.get_axes())
cax7 = divider7.append_axes("right", size="5%", pad=0.05)
f7cb=pl.colorbar(f7im,cax=cax7)
pl.savefig('/tmp/greensconvolution_compare_diff_accel.png',bbox_inches='tight')


pl.figure(8)
pl.clf()
f8im=pl.imshow(splitimg2,vmin=0.8,vmax=1.65)
pl.title('L: Finite difference; R: Simple Green\'s function (log10)')
pl.xlabel('X position')
pl.ylabel('log10(time)')
divider8=make_axes_locatable(f8im.get_axes())
cax8 = divider8.append_axes("right", size="5%", pad=0.05)
f8cb=pl.colorbar(f8im,cax=cax8)
pl.savefig('/tmp/greensconvolution_compare_lr_accelsimple.png',bbox_inches='tight')

pl.figure(9)
pl.clf()
f9im=pl.imshow(fdimage-sgfimage,vmin=-.04,vmax=.04)
pl.title('log10(finite difference)-log10(Simple Green\'s function)')
pl.xlabel('X position')
pl.ylabel('log10(time)')
divider9=make_axes_locatable(f9im.get_axes())
cax9 = divider9.append_axes("right", size="5%", pad=0.05)
f9cb=pl.colorbar(f9im,cax=cax9)
pl.savefig('/tmp/greensconvolution_compare_diff_accelsimple.png',bbox_inches='tight')
pl.show()
