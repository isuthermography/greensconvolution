# Comparison involving thin insulating layer
import sys
import os
import getpass

from greensconvolution.greensconvolution_fast import greensconvolution_integrate as greensconvolution_integrate_new
from greensconvolution.greensconvolution_calc import read_greensconvolution

import numpy as np
numpy=np
import pylab as pl
import matplotlib
import copy

import datetime

# Fix missing timedelta.total_seconds in Python < 2.7, based on http://stackoverflow.com/questions/3318348/how-can-i-extend-pythons-datetime-datetime-with-my-own-methods/14214646#14214646
import ctypes as c


matplotlib.rcParams.update({'font.size': 16})

#gc_kernel="opencl_interpolator" # greensconvolution kernel to use
#gc_kernel="openmp_interpolator" 
#gc_kernel="opencl_quadpack" 
gc_kernel="opencl_simplegaussquad" 


# ... compatibility, since this was written for the old api
def greensconvolution_integrate(greensconvolution_params,zvec,rvec,tvec,k,rho,cp):
    # zvec and rvec should have the same length, and the result will
    # be integrated over them
    # zvec is depths from surface (initial propagation)
    # rvec is distance of measurement from scatterer (scattered propagation)
    # tvec may have a different length and the result will be an array

    if zvec.shape[0]==0:
        return np.zeros(tvec.shape[0],dtype='f')
    
    return greensconvolution_integrate_new(greensconvolution_params,zvec.astype(np.float32).reshape(zvec.shape[0],1),rvec.astype(np.float32).reshape(rvec.shape[0],1),tvec.astype(np.float32).reshape(1,tvec.shape[0]),k,rho,cp,1.0,(0,),kernel=gc_kernel)

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

def build_gi_grid(dz,nz,dy,ly,dx,lx):
    z_bnd=np.arange(nz+1,dtype='d')*dz  # z boundary starts at zero
    #
    ny=int(round(ly/dy))
    nx=int(round(lx/dx))
    #
    y_bnd=np.arange(ny+1,dtype='d')*dy
    x_bnd=np.arange(nx+1,dtype='d')*dx
    #
    # Create x,y,z element center grid
    z=z_bnd[:-1]+dz/2.0
    y=y_bnd[:-1]+dy/2.0
    x=x_bnd[:-1]+dx/2.0
    #
    # Create 3d meshgrids indicating z boundary location
    # for all x,y center positions
    #
    # Voxel at i, j, k is has x boundarys at x_bnd[i] and x_bnd[i+1],
    # centered at y[k],z[k]
    # Same voxel has y boundaries at y_bnd[j] and y_bnd[j+1] which
    # are centered at x=x[i] and z=z[k]
    #
    #
    # create 3d meshgrids indicating element centers
    # print z.shape,y.shape,x.shape
    (zgrid,ygrid,xgrid) = np.meshgrid(z,y,x,indexing='ij')
    #
    return (ny,nx,
            z,y,x,
            zgrid,ygrid,xgrid,
            z_bnd,y_bnd,x_bnd)


flash_energy=10e3 # J/m^2

calc_greensfcn=True
calc_greensfcn_accel=True
calc_greensfcn_dumbaccel=False
calc_heatsim2=True
load_comsol=False #True


lamina_thickness = .2e-3
dz=lamina_thickness
dy=1e-3
dx=1e-3

# Create x,y,z voxel center coords
nz=15

(ny,nx,
 z,y,x,
 zgrid,ygrid,xgrid,
 z_bnd,y_bnd,x_bnd) = build_gi_grid(dz,nz,
                                    dy,38.0e-3,
                                    dx,36.0e-3)


z_thick=nz*dz

dz_refinement=3.0
nz=z.shape[0]
nz_refine=int(nz*dz_refinement)


z_bnd_refine=np.arange(nz_refine+1,dtype='d')*dz/dz_refinement  # z boundary starts at zero
z_refine=z_bnd_refine[:-1]+dz/(dz_refinement*2.0)

(zgrid_refine,ygrid,xgrid) = np.meshgrid(z_refine,y,x,indexing='ij')

(z_bnd_z_refine,z_bnd_y_refine,z_bnd_x_refine)=np.meshgrid(z_bnd_refine,y,x,indexing='ij')
(y_bnd_z_refine,y_bnd_y_refine,y_bnd_x_refine)=np.meshgrid(z_refine,y_bnd,x,indexing='ij')
(x_bnd_z_refine,x_bnd_y_refine,x_bnd_x_refine)=np.meshgrid(z_refine,y,x_bnd,indexing='ij')


#t0=0.01
# t0 bumped up because finite difference doesn't work too well
# at short times
t0=.07
dt=1.0/30.0
# nt=250
nt=1800

trange=t0+np.arange(nt,dtype='d')*dt



#measx=3e-3
#measy=3e-3
measx=11e-3
measy=18e-3
#measx=0.0e-3
#measy=0.0e-3


measi=np.argmin(abs(x-measx))
measj=np.argmin(abs(y-measy))



barrier_min_z=2.5e-3 #0.8e-3; 
#barrier_min_z=z_thick 
barrier_min_x=3e-3
barrier_max_x=20e-3
barrier_min_y=12.e-3
barrier_max_y=24.e-3

# define materials:
composite_k=.138 # W/m/deg K
composite_rho=1.57e3 # W/m/deg K
composite_c=730 # J/kg/deg K

barrier_k=0 # W/m/deg K
barrier_rho=1.57e3 # W/m/deg K
barrier_c=730 # J/kg/deg K

thininsulatinglayerdepth=2*dz
thininsulatinglayer_min_x=0e-3
thininsulatinglayer_max_x=50e-3
thininsulatinglayer_min_y=0e-3
thininsulatinglayer_max_y=50e-3


#thininsulatinglayer_min_x=16e-3
#thininsulatinglayer_max_x=32e-3
#thininsulatinglayer_min_y=15e-3
#thininsulatinglayer_max_y=30e-3

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
        # boundary 2: thininsulatinglayer
        # We want a thin insulating layer that halves 
        # the effective thermal effusivity of the remainder of the 
        # material 
        # e=sqrt(k rho c) has units of Joules/(m^2*deg K*sqrt(s))
        # effusivities are probably e1e2/(e1+e2) form
        # so we want e1e2/(e1+e2)=e2/2
        # e1/(e1+e2)=1/2
        # e1=e2
        # thin insulating layer coefficient is Joules/(m^2*deg K)
        # so we get this from effusivity * sqrt(characteristic time)
        # sqrt(kz*rho*c)*sqrt(thininsulatinglayerdepth**2/(np.pi*(kz/(rho*c))))
        # Simplifying: 
        #  rho*c*thininsulatinglayerdepth/np.sqrt(np.pi)
        (heatsim2.boundary_thininsulatinglayer,composite_rho*composite_c*thininsulatinglayerdepth/np.sqrt(np.pi))
)

volumetric=(  # on material grid
    # 0: nothing
    (heatsim2.NO_SOURCE,),
    #1: impulse source @ t=0
    (heatsim2.IMPULSE_SOURCE,0.0,flash_energy/(dz/dz_refinement)), # t (sec), Energy J/m^2
)


# initialize all elements to zero
(material_elements,
 boundary_z_elements,
 boundary_y_elements,
 boundary_x_elements,
 volumetric_elements)=heatsim2.zero_elements(nz*dz_refinement,ny,nx) 


# define nonzero material elements

material_elements[(zgrid_refine >= barrier_min_z) &
                  (ygrid >= barrier_min_y) &
                  (ygrid <= barrier_max_y) &
                  (xgrid >= barrier_min_x) &
                  (xgrid <= barrier_max_x)]=1 # material 1: barrier

volumetric_elements[0,:,:]=1  # set flash source (for heatsim2)


# set edges to insulating
boundary_x_elements[:,:,0]=1 # insulating
boundary_x_elements[:,:,-1]=1 # insulating
boundary_y_elements[:,0,:]=1 # insulating
boundary_y_elements[:,-1,:]=1 # insulating
boundary_z_elements[0,:,:]=1 # insulating
boundary_z_elements[-1,:,:]=1 # insulating

# add thin insulating layer

boundary_z_elements[ (z_bnd_x_refine > thininsulatinglayer_min_x) &
                     (z_bnd_x_refine < thininsulatinglayer_max_x) &
                     (z_bnd_y_refine > thininsulatinglayer_min_y) &
                     (z_bnd_y_refine < thininsulatinglayer_max_y) &
                     (z_bnd_z_refine==z_bnd_refine[(np.abs(z_bnd_refine-thininsulatinglayerdepth)).argmin()])] = 2

# set boundaries of barrier to insulating
boundary_z_elements[ (z_bnd_x_refine > barrier_min_x) &
                     (z_bnd_x_refine < barrier_max_x) &
                     (z_bnd_y_refine > barrier_min_y) &
                     (z_bnd_y_refine < barrier_max_y) &
                     (z_bnd_z_refine==z_bnd_refine[(np.abs(z_bnd_refine-barrier_min_z)).argmin()])]=1 #

boundary_x_elements[ ((x_bnd_x_refine == x_bnd[np.abs(x_bnd-barrier_min_x).argmin()])|
                      (x_bnd_x_refine == x_bnd[np.abs(x_bnd-barrier_max_x).argmin()])) &
                     (x_bnd_y_refine > barrier_min_y) &
                     (x_bnd_y_refine < barrier_max_y) &
                     (x_bnd_z_refine > barrier_min_z)]=1

boundary_y_elements[ ((y_bnd_y_refine == y_bnd[np.abs(y_bnd-barrier_min_y).argmin()])|
                      (y_bnd_y_refine == y_bnd[np.abs(y_bnd-barrier_max_y).argmin()]))&
                     (y_bnd_x_refine > barrier_min_x) &
                     (y_bnd_x_refine < barrier_max_x) &
                     (y_bnd_z_refine > barrier_min_z)]=1




t0=0.0
tf=20.0


t_heatsim2=dt/2.0+np.arange(trange[-1]//dt+1)*dt
nt_heatsim2=t_heatsim2.shape[0]
dt_heatsim2=dt


greensconvolution_params=read_greensconvolution()



(ymat,xmat)=np.meshgrid(y,x,indexing='ij')
zmat=np.ones((ny,nx),dtype='d')*z_thick
zmat[ (xmat > barrier_min_x) & (xmat < barrier_max_x) &
      (ymat > barrier_min_y) & (ymat < barrier_max_y)]=barrier_min_z
zvec=np.reshape(zmat,np.prod(zmat.shape))
insulatinglayermap=np.zeros((ny,nx),dtype=np.bool_)
insulatinglayermap[(xmat > thininsulatinglayer_min_x) &
                   (xmat < thininsulatinglayer_max_x) &
                   (ymat > thininsulatinglayer_min_y) &
                   (ymat < thininsulatinglayer_max_y) &
                   (zmat > thininsulatinglayerdepth)]=True
insulatinglayervec=insulatinglayermap.reshape(np.prod(insulatinglayermap.shape))


# WARNING: calc_greensfcn doesn't work very well near the edge of the
# domain because the images of the back surface in the domain wall
# are ignored. calc_greensfcn_accel works better because the inclusion
# of the default back surface acts as an infinite back surface. 
if calc_greensfcn:
    Tg=np.zeros((nt_heatsim2+1,ny,nx),dtype='d')
    Tg[1:,::]=((flash_energy/(composite_rho*composite_c))/np.sqrt(np.pi*(composite_k/(composite_rho*composite_c))*(t_heatsim2+dt_heatsim2/2.0))).reshape((nt_heatsim2,1,1))

    for jidx in range(ny):
        print("j=%d/%d" % (jidx,ny))
        for iidx in range(nx):
            # print("i=%d/%d" % (iidx,nx))
            #if iidx!=measi or jidx!=measj:
            #    continue
        
            dxmat=x[iidx]-xmat
            dymat=y[jidx]-ymat
            rmat=np.sqrt(dxmat**2.0+dymat**2.0+zmat**2.0)
            rvec=np.reshape(rmat,np.prod(rmat.shape))

            rmat_extraterm=np.sqrt(dxmat**2.0+dymat**2.0+(zmat*3.0)**2.0) # for extra image sources of barrier
            rvec_extraterm=np.reshape(rmat_extraterm,np.prod(rmat_extraterm.shape))


            zmat_insulatinglayer=np.ones((ny,nx),dtype='d')*thininsulatinglayerdepth
            zvec_insulatinglayer=np.reshape(zmat_insulatinglayer,np.prod(zmat_insulatinglayer.shape))
            rmat_insulatinglayer=np.sqrt(dxmat**2.0+dymat**2.0+zmat_insulatinglayer**2.0)
            rvec_insulatinglayer=np.reshape(rmat_insulatinglayer,np.prod(rmat_insulatinglayer.shape))

            rmat_extraterm_insulatinglayer=np.sqrt(dxmat**2.0+dymat**2.0+(zmat_insulatinglayer*3.0)**2.0)
            rvec_extraterm_insulatinglayer=np.reshape(rmat_extraterm_insulatinglayer,np.prod(rmat_extraterm_insulatinglayer.shape))

            
            
            # WARNING: 2.0 in leading coefficient here is a fudge factor!... where did we drop it???
            # Answer: We didn't. There's an image source reflected in the flash
            # plane required to satisfy the no-flow boundary condition on the
            # flash plane.
            # regular + image sources behind an insulating layer (weighted 0.5^2)
            
            Tg[1:,jidx,iidx]+=0.25*dx*dy*flash_energy*2.0*(greensconvolution_integrate(greensconvolution_params,zvec[insulatinglayervec],rvec[insulatinglayervec],t_heatsim2+dt_heatsim2/2.0,composite_k,composite_rho,composite_c)+greensconvolution_integrate(greensconvolution_params,zvec[insulatinglayervec],rvec_extraterm[insulatinglayervec],t_heatsim2+dt_heatsim2/2.0,composite_k,composite_rho,composite_c))

            # regular + image sources of an insulating layer (weighted 0.5)
            # image sources of layer temporarily zeroed.
            Tg[1:,jidx,iidx]+=0.5*dx*dy*flash_energy*2.0*(greensconvolution_integrate(greensconvolution_params,zvec_insulatinglayer[insulatinglayervec],rvec_insulatinglayer[insulatinglayervec],t_heatsim2+dt_heatsim2/2.0,composite_k,composite_rho,composite_c)+0.0*greensconvolution_integrate(greensconvolution_params,zvec_insulatinglayer[insulatinglayervec],rvec_extraterm_insulatinglayer[insulatinglayervec],t_heatsim2+dt_heatsim2/2.0,composite_k,composite_rho,composite_c))

            # regular + image sources not behind an insulating layer
            Tg[1:,jidx,iidx]+=dx*dy*flash_energy*2.0*(greensconvolution_integrate(greensconvolution_params,zvec[~insulatinglayervec],rvec[~insulatinglayervec],t_heatsim2+dt_heatsim2/2.0,composite_k,composite_rho,composite_c)+greensconvolution_integrate(greensconvolution_params,zvec[~insulatinglayervec],rvec_extraterm[~insulatinglayervec],t_heatsim2+dt_heatsim2/2.0,composite_k,composite_rho,composite_c))

            

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
    pass



if calc_greensfcn_dumbaccel:
    # accelerate green's function but neglect change to back face
    (ymat,xmat)=np.meshgrid(y,x,indexing='ij')
    zmat_accel=np.ones((ny,nx),dtype='d')*z_thick
    barrier_location=((xmat > barrier_min_x) & (xmat < barrier_max_x) &
                      (ymat > barrier_min_y) & (ymat < barrier_max_y))
    zmat_accel[barrier_location]=barrier_min_z
    zvec_accel=np.reshape(zmat_accel[barrier_location],np.count_nonzero(barrier_location))
    
    
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
            rvec=np.reshape(rmat[barrier_location],np.count_nonzero(barrier_location))

            rmat_extraterm=np.sqrt(dxmat**2.0+dymat**2.0+(zmat*3)**2.0) # for extra image sources of barrier
            rvec_extraterm=np.reshape(rmat_extraterm[barrier_location],np.count_nonzero(barrier_location))
            
            
            # WARNING: 2.0 in leading coefficient here is a fudge factor!... where did we drop it???
            # Answer: We didn't. There's an image source reflected in the flash
            # plane required to satisfy the no-flow boundary condition on the
            # flash plane. 
            Tg_dumbaccel[1:,jidx,iidx]+=dx*dy*flash_energy*2.0*(
                greensconvolution_integrate(greensconvolution_params,zvec_accel,rvec,t_heatsim2+dt_heatsim2/2.0,composite_k,composite_rho,composite_c)+
                greensconvolution_integrate(greensconvolution_params,zvec_accel,rvec_extraterm,t_heatsim2+dt_heatsim2/2.0,composite_k,composite_rho,composite_c))
            
            pass
        pass
    pass

                                    
if calc_heatsim2:
    hs2_start=datetime.datetime.now()
    (ADI_params,ADI_steps)=heatsim2.setup(z[0],y[0],x[0],
                                          dz/dz_refinement,dy,dx,
                                          nz*dz_refinement,ny,nx,
                                          dt_heatsim2,
                                          materials,
                                          boundaries,
                                          volumetric,
                                          material_elements,
                                          boundary_z_elements,
                                          boundary_y_elements,
                                          boundary_x_elements,
                                          volumetric_elements)
    
    
    T=np.zeros((nt_heatsim2+1,nz*dz_refinement,ny,nx),dtype='d')
    
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

if calc_greensfcn_dumbaccel:
    plotargs.extend([t_heatsim2+dt_heatsim2/2.0,Tg_dumbaccel[1:,measj,measi],'o-'])
    legendargs.append('Integral of Green\'s functions (dumb accelerated)')
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
pl.axis([1e-1,1e2,1,100])
pl.savefig("/tmp/gct2.png",dpi=300)
pl.show()


pl.figure(2)
pl.clf()
pl.loglog(*plotargs,markersize=8,linewidth=4,markeredgewidth=1)
pl.legend(legendargs,fontsize=12)
pl.xlabel('Time (s)')
pl.ylabel('Temperature (K)')
pl.grid()
pl.axis([7,100,2.2,5.6])
pl.savefig("/tmp/gct2_zoom.png",dpi=300)
pl.show()

# build log10(t) vs x images
logtrange=np.arange(-0.8,1.2,0.1,dtype='d')
ypos=0.0005
yidx=np.argmin(abs(y-.0005))

fdimage=np.zeros((logtrange.shape[0],x.shape[0]),dtype='d')
gfimage=np.zeros((logtrange.shape[0],x.shape[0]),dtype='d')
agfimage=np.zeros((logtrange.shape[0],x.shape[0]),dtype='d')
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

    nextgfvalues=np.log10(Tg[nextidx,yidx,:])
    prevgfvalues=np.log10(Tg[previdx,yidx,:])

    interpgfvalues=prevgfvalues + (nextgfvalues-prevgfvalues)*((logtval-prevlogt)/(nextlogt-prevlogt))

    nextagfvalues=np.log10(Tg_accel[nextidx,yidx,:])
    prevagfvalues=np.log10(Tg_accel[previdx,yidx,:])

    interpagfvalues=prevagfvalues + (nextagfvalues-prevagfvalues)*((logtval-prevlogt)/(nextlogt-prevlogt))

    
    fdimage[logtcnt,:]=interphs2values
    gfimage[logtcnt,:]=interpgfvalues
    agfimage[logtcnt,:]=interpagfvalues
    pass

f3=pl.figure(3)
pl.clf()
f3ax=pl.imshow(fdimage)
f3ax.get_axes().set_aspect(1.8)
pl.title('Finite difference (log10)')
pl.xlabel('X position')
pl.ylabel('log10(time)')
f3cb=pl.colorbar()
pl.savefig('/tmp/finitediff.png')

f4=pl.figure(4)
pl.clf()
f4ax=pl.imshow(gfimage)
f4ax.get_axes().set_aspect(1.8)
pl.title('Green\'s function (log10)')
pl.xlabel('X position')
pl.ylabel('log10(time)')
f4cb=pl.colorbar()
pl.savefig('/tmp/greenfn.png')
#f4cb.ax.set_aspect(1.9)

pl.figure(5)
pl.clf()
pl.imshow(agfimage)
pl.title('Accelerated Green\'s function (log10)')
pl.xlabel('X position')
pl.ylabel('log10(time)')
pl.colorbar()


xsplit=0.0
xsplitidx=np.argmin(abs(x_bnd-xsplit))

splitimg=np.concatenate((fdimage[:,:xsplit],agfimage[:,xsplit:]),axis=1)

pl.figure(6)
pl.clf()
pl.imshow(splitimg)
pl.title('Left half: Finite difference; right half: accelerated Green\'s function (log10)')
pl.xlabel('X position')
pl.ylabel('log10(time)')
pl.colorbar()

pl.figure(7)
pl.clf()
pl.imshow(fdimage-agfimage)
pl.title('Difference between log10(finite difference) and log10(Accelerated Green\'s function)')
pl.xlabel('X position')
pl.ylabel('log10(time)')
pl.colorbar()
