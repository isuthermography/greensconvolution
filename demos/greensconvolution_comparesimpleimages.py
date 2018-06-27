# Test the concept of using a hybrid greensconvolution+method of images
# where the first term is from greensconvolution and the rest
# are from method of images.

import sys
import os

import greensconvolution

from greensconvolution.greensconvolution_fast import greensconvolution_integrate_anisotropic,greensconvolution_image_sources
from greensconvolution.greensconvolution_calc import read_greensconvolution

import matplotlib
import numpy as np
numpy=np
import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

import datetime

matplotlib.rcParams.update({'font.size': 16})

greensconvolution_params=read_greensconvolution()


dy=0.5e-3
dx=0.5e-3
dt=0.05
y=np.arange(0,45e-3,dy,dtype='f')
x=np.arange(0,40e-3,dx,dtype='f')
t=np.arange(dt,10,dt,dtype='f')

nt=t.shape[0]
ny=y.shape[0]
nx=x.shape[0]

(ygrid,xgrid)=np.meshgrid(y,x,indexing="ij")

h=2e-3

kz=0.819485
ky=3.7934
kx=3.794
rho=1555.0
c=850.0

measx=20.0e-3
measy=22.1e-3

# calculationg grid: t,y,x,imageno
imageno=np.arange(1.0,20.0,2.0,dtype='f')
ni=imageno.shape[0]

gc_curve=1.0/(rho*c)/np.sqrt(np.pi*kz/(rho*c)*t) + greensconvolution_integrate_anisotropic(greensconvolution_params,h*np.ones((1,1,1,1),dtype='f'),(np.sqrt((ygrid.reshape(1,ny,nx,1)-measy)**2.0*(kz/ky) + (xgrid.reshape(1,ny,nx,1)-measx)**2.0*(kz/kx) + (h*imageno.reshape(1,1,1,ni))**2.0)).astype(np.float32),t.reshape(nt,1,1,1),kz,ky,kx,rho,c,dy*dx*2.0,(1,2,3,))

# why dx*dy*0.5???
is_curve=1.0/(rho*c)/np.sqrt(np.pi*kz/(rho*c)*t) + greensconvolution_image_sources(greensconvolution_params,(np.sqrt((ygrid.reshape(1,ny,nx)-measy)**2.0*(kz/ky) + (xgrid.reshape(1,ny,nx)-measx)**2.0*(kz/kx))).astype(np.float32),t.reshape(nt,1,1).astype(np.float32),(h*(imageno+1)).astype(np.float32),kz,rho,c,dy*dx*2.0*kz/np.sqrt(kx*ky),(1,2))


# why dx*dy*0.5???
is_curve2=1.0/(rho*c)/np.sqrt(np.pi*kz/(rho*c)*t) + (((dy*dx*2.0*kz/np.sqrt(kx*ky))*((2.0/(rho*c))/((4.0*np.pi*(kz/(rho*c))*t.reshape(nt,1,1,1))**(3.0/2.0))))*(np.exp(-((ygrid.reshape(1,ny,nx,1)-measy)**2.0*(kz/ky) + (xgrid.reshape(1,ny,nx,1)-measx)**2.0*(kz/kx)+(h*(imageno.reshape(1,1,1,ni)+1))**2.0)/(4.0*(kz/(rho*c))*t.reshape(nt,1,1,1))))).sum((1,2,3))


is_curve3=greensconvolution_image_sources(greensconvolution_params,(np.sqrt((ygrid.reshape(1,ny,nx)-measy)**2.0*(kz/ky) + (xgrid.reshape(1,ny,nx)-measx)**2.0*(kz/kx))).astype(np.float32),t.reshape(nt,1,1).astype(np.float32),(h*(imageno+1)).astype(np.float32),kz,rho,c,dy*dx*2.0*kz/np.sqrt(kx*ky),(1,2))

# why dx*dy*0.5???
combo_curve=1.0/(rho*c)/np.sqrt(np.pi*kz/(rho*c)*t) + greensconvolution_integrate_anisotropic(greensconvolution_params,h*np.ones((1,1,1),dtype='f'),(np.sqrt((ygrid.reshape(1,ny,nx)-measy)**2.0*(kz/ky) + (xgrid.reshape(1,ny,nx)-measx)**2.0*(kz/kx) + (h*imageno[0])**2.0)).astype(np.float32),t.reshape(nt,1,1),kz,ky,kx,rho,c,dy*dx*2.0,(1,2,)) + greensconvolution_image_sources(greensconvolution_params,(np.sqrt((ygrid.reshape(1,ny,nx)-measy)**2.0*(kz/ky) + (xgrid.reshape(1,ny,nx)-measx)**2.0*(kz/kx))).astype(np.float32),t.reshape(nt,1,1).astype(np.float32),(h*(imageno+3)).astype(np.float32),kz,rho,c,dy*dx*2.0*kz/np.sqrt(kx*ky),(1,2))


pl.figure(1)
pl.clf()
pl.plot(t,gc_curve,'-',t,is_curve,'-',t,is_curve2,'-',t,combo_curve,'-')

pl.figure(2)
pl.clf()
pl.loglog(t,gc_curve,'-',t,is_curve,'-',t,is_curve2,'-',t,combo_curve,'-')

pl.show()
