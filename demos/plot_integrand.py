import numpy as np
from matplotlib import pyplot as pl

integrand = lambda x,v,a: x**(-1.5)*(1.0-x)**(-1.5)*np.exp(-(1.0+a*x)/(v*x*(1.0-x)))
# where a = c^2 -1

vrange=10**np.arange(-5,4,.1)

# in greensinversion c always at least 1 because
# distance from reflector back to surface
# must be at least depth of reflector
crange=10**np.arange(0,1,.1)  # Note: Convergence trouble for small values of c... fortunately we don't care because c < 1 means 3d propagation is much shorter than 1D distance
xrange=np.arange(0.001,.999,.001)

arange = crange**2.0 - 1.0

(v_bc,a_bc,x_bc) = np.broadcast_arrays(vrange[:,np.newaxis,np.newaxis],arange[np.newaxis,:,np.newaxis],xrange[np.newaxis,np.newaxis,:])

integrand_eval=integrand(x_bc,v_bc,a_bc)

# plot superposition of integrands(v) at range of c's
for cidx in range(crange.shape[0]):
    pl.figure(cidx)
    pl.clf()
    pl.plot(xrange,integrand_eval[:,cidx,:].T/np.max(integrand_eval[:,cidx,:],1),'-')
    pl.title('c=%f' % (crange[cidx]))
    pass

# at c=1.0 (i.e. a=0), plot raw integrands
cidx=0
for vidx in range(vrange.shape[0]):
    pl.figure(crange.shape[0]+vidx)
    pl.clf()
    pl.plot(xrange,integrand_eval[vidx,cidx,:]/np.max(integrand_eval[vidx,cidx,:]),'-')
    pl.title('v=%f c=%f' % (vrange[vidx],crange[cidx]))
    pass
# @ c=0.0, v peaks around x=0.5 for v < ~3, peaks at boundaries 0 and 1
# for v > 3
# v represents normalized time 4alphazt/z^2

# Where does the integrand peak?
# Take it's derivative with respect to x
# (1/((1-x)^(3/2) * x^(3/2))) * exp(- (1+ax)/(vx(1-x)))
# (1-x)^(-3/2) * x^(-3/2) * exp((-1-ax)/(vx - vx^2))

# (3/2)(1-x)^(-5/2) * x^(-3/2) * exp(-(1+ax)/(vx(1-x))) +
#   (1-x)^(-3/2) * (-3/2)x^(-5/2) * exp(- (1+ax)/(vx(1-x))) +
#   (1-x)^(-3/2) * x^(-3/2) * exp(- (1+ax)/(vx(1-x))) * [ (-a)*(vx-vx^2) + (1+ax)*(v-2vx) ]/((vx)^2*(1-x)^2)  = 0 at the peak
#   ... the exponential is the same factor in all terms and never equal to zero except in limits. Divide it out
# (3/2)(1-x)^(-5/2) * x^(-3/2) +
#   (1-x)^(-3/2) * (-3/2)x^(-5/2) +
#   (1-x)^(-3/2) * x^(-3/2) * [ (-a)*(vx-vx^2) + (1+ax)*(v-2vx) ]/((vx)^2*(1-x)^2)  = 0
#  Multiply through by (1-x)^(5/2) and x^(5/2)
# (3/2)*x +
#   (1-x)*(-3/2) +
#   (1-x)*x*[ (-a)*(vx-vx^2) + (1+ax)*(v-2vx) ]/((vx)^2*(1-x)^2)  = 0
# Simplify...
# (3/2)*x +
#   -(3/2- (3/2)x) +
#   (1-x)*x*[ (-a)*(vx-vx^2) + (1+ax)*(v-2vx) ]/((vx)^2*(1-x)^2)  = 0
# ... 
# 3*x - 3/2 +
#   (1-x)*x*[ (-a)*(vx-vx^2) + (1+ax)*(v-2vx) ]/((vx)^2*(1-x)^2)  = 0
# multiply through by vx^2 * (1-x)^2  -- note only one factor of v
# (3*v*x^3)(1-x)^2 - ((3/2)vx^2)*(1-x)^2 +
#   (1-x)*x*[ (-a)*(x-x^2) + (1+ax)*(1-2x) ]  = 0
# ... divide by x and (1-x)
# (3*v*x^2)(1-x) - ((3/2)vx)*(1-x) +
#   (-a)*(x-x^2) + (1+ax)*(1-2x)  = 0
# group...
# (3*v*x^2)(1-x) - ((3/2)vx)*(1-x) +
#   (-a*x)*(1-x) + (1+ax)*(1-2x)  = 0
#       ... cubic equation in x
# i.e. 3vx^2 - 3vx^3 - (3/2)vx + (3/2)vx^2
#        -ax + ax^2 + 1 - 2x + ax - 2ax^2 = 0
# i.e. -3vx^3 + (3v + (3/2)v + a -2a)x^2 + ((-3/2)v - a -2 +a)x +1 = 0
# i.e. -3vx^3 + ((9/2)v -a)x^2 -((3/2)v+2)x + 1 = 0
pl.figure(crange.shape[0]+vrange.shape[0])
pl.clf()


#(v_bc2,a_bc2) = np.broadcast_arrays(vrange[:,np.newaxis],arange[np.newaxis,:])
polyroots = np.ones((vrange.shape[0],arange.shape[0]),dtype='d')*np.NaN

for vcnt in range(vrange.shape[0]):
    for acnt in range(arange.shape[0]):
        gotroots=np.roots((-3.0*vrange[vcnt],4.5*vrange[vcnt]-arange[acnt],-1.5*vrange[vcnt]-2.0,1.0))
        usefulroots = gotroots[(gotroots.imag==0) & (gotroots.real > 0)]

        if len(usefulroots)==1:
            assert(usefulroots[0].imag==0.0)
            polyroots[vcnt,acnt]=usefulroots[0].real
            pass
        pass
    pass

pl.imshow(polyroots,origin='lower',extent=(np.log(crange[0]),np.log(crange[-1]),np.log(vrange[0]),np.log(vrange[-1])))
pl.colorbar()
pl.xlabel('ln(c)')
pl.ylabel('ln(v)')
pl.title('x position of peak of integrand')

pl.show()
