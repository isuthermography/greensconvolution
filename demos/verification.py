import numpy as np
import scipy
import scipy.integrate

from greensconvolution.greensconvolution_fast import greensconvolution_integrate_anisotropic, greensconvolution_image_sources, greensconvolution_greensfcn_curved
from greensconvolution.greensconvolution_calc import read_greensconvolution

try:
    # Reference curved_laminate_final_2d if available
    from curved_laminate_final_2d import nondimensionalize as curved_gf_nondim
    from curved_laminate_final_2d import evaluate as curved_gf_eval
    pass
except ImportError:
    curved_gf_eval=None
    pass


# Test case for flat and curved cases

rho=float(1.555e3) # kg/m^3
c=float(850.0) # J/(kg* deg K)
alphaz=float(.62e-6) # best evaluation based on ishmael:/dataawareness/NASA_Modeling/Thermal_Diffusivity_Test_Data/thermalconductivity.ods 7/10/16 based on flash through-thickness and cross-fiber in-plane tests
alphaxy=float(2.87e-6) # best evaluation based on ishmael:/dataawareness/NASA_Modeling/Thermal_Diffusivity_Test_Data/thermalconductivity.ods 7/10/16 based on 0/90 and quasi-isotropic layups

concave_curv_x = 1.0/(.125*25.4e-3) # uniform .125 inch radius of curvature along x at first (top) layer
convex_curv_x = -1.0/(.125*25.4e-3) # uniform .125 inch radius of curvature along x at first (top) layer

source_depth=6.0*8.05e-3/(3.0*16.0)  # meters
source_energy=100.0 # Joules for a point source (verification is 3D whereas most of the curved_laminate stuff is 2D)

t=1.5 # seconds
x = 1e-3 # mm 
y = 0.0 # mm

alphaxyz = (alphaz*alphaxy**2)**(1.0/3.0)
v=4.0*alphaz*t/(source_depth**2.0)

concave_w = concave_curv_x*source_depth*np.sqrt(np.pi)/8.0
convex_w = convex_curv_x*source_depth*np.sqrt(np.pi)/8.0


def inner_sin_sq(x):
    if np.abs(x) <= np.pi/4.0:
        return np.sin(x)**2.0
    else:
        return 0.25+(np.abs(x)-np.pi/4.0+0.5)**2.0
    pass


# cedilla (flat)
ced = np.sqrt( (x**2.0)*(alphaz/alphaxy) + (y**2.0)*(alphaz/alphaxy) + source_depth**2.0)/np.abs(source_depth)

# cedilla (concave curved) from eq. 38 in greensfcn_doc.pdf
concave_theta = concave_curv_x * x
if abs(concave_theta) > np.pi/2.0:
    concave_theta=np.pi/2.0
    pass

concave_deceleration=(1.0+(1.0/12.0)*(concave_theta**2.0))**(2.0*(1.0 - source_depth/(np.abs(x)*np.abs(concave_theta/2.0 + (concave_theta**3.0)/24.0))))
if concave_deceleration < 1.0:
    concave_deceleration=1.0
    pass

concave_ced = np.sqrt( (x**2)*(1+source_depth*concave_curv_x)*concave_deceleration*(alphaz/alphaxy) + (y**2.0)*(alphaz/alphaxy) + source_depth**2.0)/np.abs(source_depth)

# cedilla (convex curved) from eq. 38 in greensfcn_doc.pdf
convex_ced = np.sqrt( (((1.0/np.abs(convex_curv_x)) - source_depth)**2.0)*(1.0 + (np.abs(convex_curv_x)*source_depth)/(1.0-np.abs(convex_curv_x)*source_depth))*4.0*inner_sin_sq(np.abs(convex_curv_x)*x/2.0)*(alphaz/alphaxy) + (y**2.0)*(alphaz/alphaxy) + source_depth**2.0)/np.abs(source_depth)


# a
a = ced**2.0 - 1.0
concave_a = concave_ced**2.0 - 1.0
convex_a = convex_ced**2.0 - 1.0

# Direct evaluation of integral (flat case) eq. 31 from greensfcn_doc.pdf
# (dropping the leading minus sign)
flatcase = (2.0*alphaz**(3.0/2.0)/(rho*c*(np.pi**2.0)*(alphaxyz**(3.0/2.0))*(source_depth**3.0)*(v**2.0)))*scipy.integrate.quad(lambda s: (1.0/(((1.0-s)**(3.0/2.0))*(s**(3.0/2.0))))*np.exp(-(1.0+a*s)/(v*s*(1.0-s))),0.0,1.0)[0]

gc_params=read_greensconvolution()

coeff=1.0
# Using the accelerated Greensconvolution code, (also dropping the leading minus sign)
flatcase_gc_quad = greensconvolution_integrate_anisotropic(gc_params,np.array((source_depth,),dtype='f'),np.array((x,),dtype='f'),np.array((t,),dtype='f'),0.0,alphaz*rho*c,alphaxy*rho*c,alphaxy*rho*c,rho,c,coeff,np.array((),dtype='i'),avgcurvatures=None,kernel="opencl_quadpack")

flatcase_gc_interp = greensconvolution_integrate_anisotropic(gc_params,np.array((source_depth,),dtype='f'),np.array((x,),dtype='f'),np.array((t,),dtype='f'),0.0,alphaz*rho*c,alphaxy*rho*c,alphaxy*rho*c,rho,c,coeff,np.array((),dtype='i'),avgcurvatures=None,kernel="opencl_interpolator")

print("Flat Direct: %f" % (flatcase))
print("Flat GC quadpack: %f" % (flatcase_gc_quad))
print("Flat GC interpolator: %f" % (flatcase_gc_interp))

flatcase_error_gc_quad = (flatcase_gc_quad-flatcase)/flatcase
flatcase_error_gc_interp = (flatcase_gc_interp-flatcase)/flatcase

print("Flat GC quadpack error: %f%%" % (flatcase_error_gc_quad*100.0))
print("Flat GC interpolator error: %f%%" % (flatcase_error_gc_interp*100.0))

assert(abs(flatcase_error_gc_quad) < 1e-3)
assert(abs(flatcase_error_gc_interp) < 1e-3)

# Direct evaluation of integral (curved case)  eq. 40 and 43 from greensfcn_doc.pdf
# (again dropping the leading minus sign)

concave_exact = (2.0*alphaz**(3.0/2.0)/(rho*c*(np.pi**2.0)*(alphaxyz**(3.0/2.0))*(source_depth**3.0)*(v**2.0)))*scipy.integrate.quad(lambda s: (1.0/(((1.0-s)**(3.0/2.0))*(s**(3.0/2.0))))*(1.0/(1.0+concave_w*np.sqrt(v*(1.0-s))))*np.exp(-(1.0+concave_a*s)/(v*s*(1.0-s))),0.0,1.0)[0]
convex_exact = (2.0*alphaz**(3.0/2.0)/(rho*c*(np.pi**2.0)*(alphaxyz**(3.0/2.0))*(source_depth**3.0)*(v**2.0)))*scipy.integrate.quad(lambda s: (1.0/(((1.0-s)**(3.0/2.0))*(s**(3.0/2.0))))*(1.0/(1.0+convex_w*np.sqrt(v*(1.0-s))))*np.exp(-(1.0+convex_a*s)/(v*s*(1.0-s))),0.0,1.0)[0]


# WARNING: the _approx values do not include the bounding of w*sqrt(v)!!!
concave_approx = (2.0*alphaz**(3.0/2.0)/(rho*c*(np.pi**2.0)*(alphaxyz**(3.0/2.0))*(source_depth**3.0)*(v**2.0)))*(1.0/(1.0+0.8*concave_w*np.sqrt(v)))*scipy.integrate.quad(lambda s: (1.0/(((1.0-s)**(3.0/2.0))*(s**(3.0/2.0))))*np.exp(-(1.0+concave_a*s)/(v*s*(1.0-s))),0.0,1.0)[0]
convex_approx = (2.0*alphaz**(3.0/2.0)/(rho*c*(np.pi**2.0)*(alphaxyz**(3.0/2.0))*(source_depth**3.0)*(v**2.0)))*(1.0/(1.0+0.8*convex_w*np.sqrt(v)))*scipy.integrate.quad(lambda s: (1.0/(((1.0-s)**(3.0/2.0))*(s**(3.0/2.0))))*np.exp(-(1.0+convex_a*s)/(v*s*(1.0-s))),0.0,1.0)[0]

print("amplitude factor approximation error (convex): %f%%" % ((convex_approx-convex_exact)*100.0/convex_exact))
print("amplitude factor approximation error (concave): %f%%" % ((concave_approx-concave_exact)*100.0/concave_exact))

concave_gc_interp = greensconvolution_integrate_anisotropic(gc_params,np.array((source_depth,),dtype='f'),np.array((x,),dtype='f'),np.array((t,),dtype='f'),0.0,alphaz*rho*c,alphaxy*rho*c,alphaxy*rho*c,rho,c,coeff,np.array((),dtype='i'),avgcurvatures=np.array((concave_curv_x,),dtype='f'),kernel="opencl_interpolator_curved")


convex_gc_interp = greensconvolution_integrate_anisotropic(gc_params,np.array((source_depth,),dtype='f'),np.array((x,),dtype='f'),np.array((t,),dtype='f'),0.0,alphaz*rho*c,alphaxy*rho*c,alphaxy*rho*c,rho,c,coeff,np.array((),dtype='i'),avgcurvatures=np.array((convex_curv_x,),dtype='f'),kernel="opencl_interpolator_curved")

concave_error_gc_interp = (concave_gc_interp-concave_approx)/concave_approx
convex_error_gc_interp = (convex_gc_interp-convex_approx)/convex_approx

print("Concave GC interpolator error: %f%%" % (concave_error_gc_interp*100.0))
print("Convex GC interpolator error: %f%%" % (convex_error_gc_interp*100.0))

assert(abs(concave_error_gc_interp) < 1e-3)
assert(abs(convex_error_gc_interp) < 1e-3)


# Image sources...
#
# Per discussion in paper use simple image sources
# to help satisfy zero heat flow boundary condition on
# both free surface and defect (i.e. delamination) surface

# return Green's function response
#  for a sum of image sources at the same
#  lateral position but different depths.

image_source_flat = (2.0/(rho*c))*(1.0/(np.sqrt(alphaz)*alphaxy*(4.0*np.pi*t)**(3.0/2.0)))*np.exp(-(source_depth*4.0)**2.0/(4.0*alphaz*t) - x**2.0/(4.0*alphaxy*t))
image_source_flat_gc = greensconvolution_image_sources(gc_params,np.sqrt(np.array((x,),dtype='f')**2.0*(alphaz/alphaxy)),np.array((t,),dtype='f'),np.array((source_depth*4.0,),dtype='f'),alphaz*rho*c,rho,c,1.0,(),kxy=alphaxy*rho*c)

print("Image source flat = %f" % (image_source_flat))
print("Image source flat GC = %f" % (image_source_flat_gc))

image_source_flat_error=(image_source_flat_gc-image_source_flat)/image_source_flat
print("Image source flat error = %f%%" % (image_source_flat_error*100.0))
assert(abs(image_source_flat_error) < 1e-6)

# Image sources -- for these we use a simplified curved scenario Green's function based on
#  images at 4z, 6z, 8z, etc atop and underneath the free surface.
#
# We use the ExtraVolumeFactor from the original source (based on the curvature)
# We assume that the other curved surface corrections roughly
# balance each other out from the image on either side of the
# curved surface.
#
# In general, there is an extra factor of 2 representing two image sources,
# but this is supplied by the caller of greensconvolution_image_sources()
#
# NOTE: We are not bounding ExtraVolumeFactor here
image_source_concave = (4.0/(rho*c))*(1.0/(np.sqrt(alphaz)*alphaxy*(4.0*np.pi*t)**(3.0/2.0)))*(1.0/(1.0+0.25*concave_curv_x*np.sqrt(np.pi*alphaz*t)))*np.exp(-(source_depth*4.0)**2.0/(4.0*alphaz*t) - (x**2.0/(4.0*alphaxy*t)))

image_source_concave_gc = greensconvolution_image_sources(gc_params,np.sqrt(np.array((x,),dtype='f')**2.0*(alphaz/alphaxy)),np.array((t,),dtype='f'),np.array((source_depth*4.0,),dtype='f'),alphaz*rho*c,rho,c,2.0,(),avgcurvatures=np.array((concave_curv_x,),dtype='f'),kxy=alphaxy*rho*c)

image_source_concave_error = (image_source_concave_gc-image_source_concave)/image_source_concave
print("image source concave error = %f%%" % (image_source_concave_error*100.0))
assert(abs(image_source_concave_error) < 1e-6)


# Raw Green's function, in 3D
# NOTE: We are not bounding ExtraVolumeFactor here
concave_gf_eval = source_energy * (2.0/(rho*c*(4.0*np.pi)**(3.0/2.0)*np.sqrt(alphaz)*alphaxy*t**(3.0/2.0))) * (1.0/(1.0+0.25*concave_curv_x*np.sqrt(np.pi*alphaz*t))) * np.exp(-(source_depth**2.0)/(4.0*alphaz*t) - ((x**2.0)/(4.0*alphaxy*t))*(1.0+source_depth*concave_curv_x)*concave_deceleration)

convex_gf_eval = source_energy * (2.0/(rho*c*(4.0*np.pi)**(3.0/2.0)*np.sqrt(alphaz)*alphaxy*t**(3.0/2.0))) * (1.0/(1.0+0.25*convex_curv_x*np.sqrt(np.pi*alphaz*t))) * np.exp(-(source_depth**2.0)/(4.0*alphaz*t) - ((((1.0/np.abs(convex_curv_x))-source_depth)**2.0)/(4.0*alphaxy*t))*(1.0 + source_depth/((1.0/np.abs(convex_curv_x))-source_depth))*4.0*inner_sin_sq(0.5*x*convex_curv_x))


concave_gf_gc_eval = greensconvolution_greensfcn_curved(gc_params,np.array((source_energy,),dtype='f'),np.array((x,),dtype='f'),np.array((source_depth,),dtype='f'),np.array((0.0,),dtype='f'),np.array((t,),dtype='f'),alphaz*rho*c,rho,c,(),avgcurvatures=np.array((concave_curv_x,),dtype='f'),avgcrosscurvatures=np.array((0.0,),dtype='f'),ky=alphaxy*rho*c,kx=alphaxy*rho*c)

convex_gf_gc_eval = greensconvolution_greensfcn_curved(gc_params,np.array((source_energy,),dtype='f'),np.array((x,),dtype='f'),np.array((source_depth,),dtype='f'),np.array((0.0,),dtype='f'),np.array((t,),dtype='f'),alphaz*rho*c,rho,c,(),avgcurvatures=np.array((convex_curv_x,),dtype='f'),avgcrosscurvatures=np.array((0.0,),dtype='f'),ky=alphaxy*rho*c,kx=alphaxy*rho*c)

concave_gf_error = (concave_gf_gc_eval-concave_gf_eval)/concave_gf_eval
convex_gf_error = (convex_gf_gc_eval-convex_gf_eval)/convex_gf_eval

print("concave_gf_error = %f%%" % (concave_gf_error*100.0))
print("convex_gf_error = %f%%" % (convex_gf_error*100.0))

assert(abs(concave_gf_error) < 1e-6)
assert(abs(convex_gf_error) < 1e-6)

if curved_gf_eval is not None:
    # test against curved_laminate_final_2d.py if available
    
    # Note: curved_gf_eval gives the 2D response.
    # We need to multiply the 3rd dimension in here
    # to make it comparable, i.e. divide by sqrt(4*pi*alphaxy*t)

    concave_gf_params = curved_gf_nondim(source_energy,rho,c,alphaz,alphaxy,concave_curv_x,x,source_depth,t)
    concave_gf_cl_eval = (1.0/(np.sqrt(4.0*np.pi*alphaxy*t)))*curved_gf_eval(*concave_gf_params)

    convex_gf_params = curved_gf_nondim(source_energy,rho,c,alphaz,alphaxy,convex_curv_x,x,source_depth,t)
    convex_gf_cl_eval = (1.0/(np.sqrt(4.0*np.pi*alphaxy*t)))*curved_gf_eval(*convex_gf_params)

    concave_gf_cl_error = (concave_gf_cl_eval-concave_gf_eval)/concave_gf_eval
    convex_gf_cl_error = (convex_gf_cl_eval-convex_gf_eval)/convex_gf_eval
    
    print("concave_gf_cl_error = %f%%" % (concave_gf_cl_error*100.0))
    print("convex_gf_cl_error = %f%%" % (convex_gf_cl_error*100.0))

    assert(abs(concave_gf_cl_error) < 1e-6)
    assert(abs(convex_gf_cl_error) < 1e-6)
    
    pass
else:
    print("curved_laminate_final_2d not available; using internal approximation code only")
    pass
