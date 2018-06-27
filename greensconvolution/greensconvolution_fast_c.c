#ifndef __OPENCL_VERSION__
/* only for non-opencl */
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

#include "greensconvolution_fast_c.h"

#define OPENCL_GLOBAL
#define OPENCL_KERNEL
#define CONSTGLOBAL static const

#define USE_OPENMP

#else /* __OPENCL_VERSION__ */

#ifndef NULL
#define NULL ((void *)0L)
#endif

#define CONSTGLOBAL __constant
#define OPENCL_GLOBAL __global
#define OPENCL_KERNEL __kernel


#endif /* __OPENCL_VERSION__ */


#ifndef TRUE
#define TRUE (!0)
#endif

#ifndef FALSE
#define FALSE (0)
#endif



// Following the logic at the top of imagesources_curved.c,
// the curved case is identical to the flat case,
// using the same leading coefficients. 
 
// The differences are:
// The empirically determined curvature-dependent coefficient. 
// Predicted_T *=  1.0/(1.0 + tc*coeffs + dc*coeff2 + tc2*coeffs**2.0 + dc2*coeff2**2 + tcdc*coeffs*coeffs + tc3*coeffs**3.0 + tc2dc*coeffs**2.0*coeff2)
// 
// and
// Predicted_T[:,concave_in_x] *= np.exp( -(a/(4.0*alphaz*Pred_t[:,:,0])*(1.0 + source_depth*Pred_kx[:,concave_in_x])*Pred_x[:,concave_in_x]**2))
//     or
// Predicted_T[:,~concave_in_x] *= np.exp( -(a/(4.0*alphaz*Pred_t[:,:,0])*(1.0 + source_depth/((1.0/Pred_kx[:,~concave_in_x])+source_depth  ))*Pred_x[:,~concave_in_x]**2))
//   (to analyze the above factors, we use the second, and also roll in the exp(-z^2/4alphat)

// Where a=alphaz/alphaxy
// coeff2 is source_depth*curvature
// Pred_x is rvec
//    .... replaces exp(-x^2/(4*alphaxy*t)  in the uncurved version 

/* Noncurved function:

integral from greensfcn_doc.tex 

(leading coefficient divided by z^3v^2) times integral from 0 to 1 of: 
doublereal funct_(doublereal *x, doublereal *v, doublereal *a) 
{
  return pow(*x,-1.5f)*pow(1.0f-(*x),-1.5f)*exp(-(1.0f+(*a)*(*x))/((*v)*(*x)*(1.0f-(*x))));
}



... How to apply the differences? 
  * The empirically determined curvature-dependent coefficient. 
    has z dependence and curvature dependence but is outside of the integral. 

    *  The exponential factor is part of the integrand and needs to be converted
    to the dimensionless variables for efficient computation. 

To do this, roll in the exp(-z^2/(4alphaz t) ) factor 
... substitute a, substitute t-tau for Pred_t, substitute z for source_depth

// Predicted_T[:,~concave_in_x] *= np.exp( -z^2/(4alphaz (t-tau)) -(1/(4.0*alphaxy*(t-tau)))*(1.0 + z/((1.0/curvature) + z))*Pred_x[:,~concave_in_x]**2)
// substitute (t-tau)=(v-u) * (z^2/(4alphaz)),  
// Predicted_T[:,~concave_in_x] *= np.exp( -z^2/(4alphaz (v-u)*z^2/4alphaz) -(1/(4.0*alphaxy*(v-u)*(z^2/(4alphaz)))*(1.0 + z/((1.0/curvature) + z  ))*Pred_x[:,~concave_in_x]**2))
// Predicted_T[:,~concave_in_x] *= np.exp( -1/(v-u) -(alphaz/(alphaxy*(v-u)*(z^2)))*(1.0 + z/((1.0/curvature) + z  ))*Pred_x[:,~concave_in_x]**2))
// Predicted_T[:,~concave_in_x] *= np.exp( -1/(v-u) -(alphaz/alphaxy)*(1/(v-u))*(1/z^2)*(1.0 + z/((1.0/curvature) + z  ))*Pred_x[:,~concave_in_x]**2))
// In this context, Pred_x represents the in-plane measurement distance 
// so cedilla^2 = (pred_x^2(alphaz/alphaxy) + z^2)/z^2  
// so cedilla^2z^2 - z^2= pred_x^2(alphaz/alphaxy)  
//    pred_x^2 = (cedilla^2 - 1)*z^2 * alphaxy/alphaz
// Predicted_T[:,~concave_in_x] *= np.exp( -1/(v-u) -(alphaz/alphaxy)*(1/(v-u))*(1/z^2)*(1.0 + z/((1.0/curvature) + z  ))*(cedilla^2-1) * z^2 * (alphaxy/alphaz) ))
// Cancel alphaz/alphaxy and z^2 
// Predicted_T[:,~concave_in_x] *= np.exp( -1/(v-u) -(1/(v-u))*(1.0 + z/((1.0/curvature) + z  ))*(cedilla^2-1) ))
// Predicted_T[:,~concave_in_x] *= np.exp( -1/(v-u) -(1/(v-u))*(1.0 + z*curvature/(1.0 + z*curvature  ))*(cedilla^2-1) ))
// Predicted_T[:,~concave_in_x] *= np.exp( -(1/(v-u))(1 + (1.0 + z*curvature/(1.0 + z*curvature  ))*(cedilla^2-1) ))
// Predicted_T[:,~concave_in_x] *= np.exp( -(1/(v-u))(1 + (cedilla^2-1 + (z*curvature/(1.0 + z*curvature))*(cedilla^2-1) )))
// Predicted_T[:,~concave_in_x] *= np.exp( -(1/(v-u))(cedilla^2 + (z*curvature/(1.0 + z*curvature))*(cedilla^2-1)))
// This replaces the np.exp( -(cedilla^2/(v-u))) factor in the flat case
// Add in rest of exponential factor from greensfcn_doc.pdf (i.e. exp(-1/u))
// Predicted_T[:,~concave_in_x] *= np.exp(-(1/u) -(1/(v-u))(cedilla^2 + (z*curvature/(1.0 + z*curvature))*(cedilla^2-1)))
//  *******
// Substitute x=u/v  as in the bottom of greensfcn_doc.pdf
// Predicted_T[:,~concave_in_x] *= np.exp(-(1/xv) -(1/(v-xv))(cedilla^2 + (z*curvature/(1.0 + z*curvature))*(cedilla^2-1)))
// Common denominator
// Predicted_T[:,~concave_in_x] *= np.exp(-((1 - x + x(cedilla^2 + (z*curvature/(1.0 + z*curvature))*(cedilla^2-1)))/(xv(1-x))))
// let a = (cedilla^2 + (z*curvature/(1.0 + z*curvature))*(cedilla^2-1))-1  # Not to be confused with the a=alphaz/alphaxy used above
// Predicted_T[:,~concave_in_x] *= np.exp(-(1 +ax)/(xv(1-x)))
// ... Which is the same as the regular Greensconvolution except for the redefinition of a!!!!! 
//  Simplify a: 
// a = cedilla^2 + (z*curvature/(1.0 + z*curvature))*cedilla^2 -(z*curvature/(1.0 + z*curvature)) -1   # Not to be confused with the a=alphaz/alphaxy used above
// a = cedilla^2(1 + (z*curvature/(1.0 + z*curvature))) - (z*curvature/(1.0 + z*curvature)) -1   # Not to be confused with the a=alphaz/alphaxy used above
// let w = (z*curvature/(1.0+z*curvature))
// a = cedilla^2(1+w) - (1+w)
// a = (cedilla^2-1)(1+w)
// NOTE:   the z*curvature in the denominator of w is only present in the convex case!


// The implementation below is in terms of cedilla and v, 
// not cedilla and a.
// Continuing above derivation from line marked as ******** 
// Predicted_T[:,~concave_in_x] *= np.exp(-(1/u) -(1/(v-u))(cedilla^2 + (z*curvature/(1.0 + z*curvature))*(cedilla^2-1)))
// let w = (z*curvature/(1.0+z*curvature))   ( convex case, i.e. curvature < 0 ) 
// or  w = (z*curvature)    (concave case, i.e. curvature > 0)
// Predicted_T[:,~concave_in_x] *= np.exp(-(1/u) -(1/(v-u))(cedilla^2 + w*(cedilla^2-1)))
//   So what had been just cedilla^2 in the flat formula is now (cedilla^2 + w*(cedilla^2-1))
//   or curved_cedilla^2 = flat_cedilla^2 + w*(flat_cedilla^2-1)
//   or curved_cedilla^2 = flat_cedilla^2 + w*flat_cedilla^2-w)
//   or curved_cedilla^2 = flat_cedilla^2(1+w) - w)
//   or curved_cedilla^2 = flat_cedilla^2(1+w) +1 -1 - w)
//   or curved_cedilla^2 = flat_cedilla^2(1+w) +1 -(1+w)
//   or curved_cedilla^2 = (flat_cedilla^2-1)(1+w) + 1
//   or curved_cedilla^2 = ((r_conductivityscaled/z)^2-1)(1+w) +1

// in the convex case (negative curvature) as the denominator of w i.e. (1+z*curvature) gets too small
// (or becomes zero or negative in the extreme case), what that is really doing 
// is making the factor (1+w) approach zero. To avoid numerical trouble and non-physical
// situations, we just force (1+w) to be positive or zero. 
 

*/


     // Coefficients for curved case , from eval_curved_laminate.py 10/25/16 02:32pm
  // THESE PROBABLY NEED TO BE RECALCULATED FOR THE SIMPLIFIED LAMINATE
  // OR DETERMINED FROM THEORY!!!
							    /*
CONSTGLOBAL float tc=0.40211414;
CONSTGLOBAL float dc=-0.00423593;
CONSTGLOBAL float tc2= -0.03041781;
CONSTGLOBAL float dc2=0.0027296;
CONSTGLOBAL float tcdc=-0.05028247;
CONSTGLOBAL float tc3=-0.00087789;
CONSTGLOBAL float  tc2dc=0.00723961;
							    */
static inline float inner_sin_sq(float x)
{
  if (fabs(x) < M_PI/4.0f) {
    return pow(sin(x),2.0f);
  } else {
    return 0.25 + pow(fabs(x)-((float)M_PI)/4.0f+0.5f,2.0f);
  }
}

 
static inline void greensconvolution_integrate_anisotropic_c_one(uint64_t itercnt,
								 OPENCL_GLOBAL const float *vrange,uint64_t nvrange, //number of rows in integraleval
								 OPENCL_GLOBAL const float *crange,uint64_t ncrange, // number of cols in integraleval
								 OPENCL_GLOBAL const float *integraleval,
								 OPENCL_GLOBAL const float *integral_dintegranddveval, // same size as integraleval
								 OPENCL_GLOBAL const float *integral_dintegranddceval, // same size as integraleval
								 OPENCL_GLOBAL const float *zvec, OPENCL_GLOBAL const uint64_t *zvecshape,OPENCL_GLOBAL const uint64_t *zvecstrides,
								 OPENCL_GLOBAL const float *xvec, OPENCL_GLOBAL const uint64_t *xvecshape,OPENCL_GLOBAL const uint64_t *xvecstrides,
								 OPENCL_GLOBAL const float *tvec, OPENCL_GLOBAL const uint64_t *tvecshape,OPENCL_GLOBAL const uint64_t *tvecstrides,
								 float yval,
								 OPENCL_GLOBAL const float *curvaturevec, OPENCL_GLOBAL const uint64_t *curvaturevecshape, OPENCL_GLOBAL const uint64_t *curvaturevecstrides,
								 OPENCL_GLOBAL const uint64_t *sumstrides, OPENCL_GLOBAL const uint64_t *shape,

								 
								 OPENCL_GLOBAL volatile float *result,OPENCL_GLOBAL const uint64_t *resultstrides,
								 float coeff,
								 OPENCL_GLOBAL const uint64_t *axissumflag,
								 uint64_t ndim,
								 float log10v0,float log10c0,float dlog10v,float dlog10c,float alphaz,float alphaxy,int8_t curvature_flag)  
{
  uint64_t zpos,tpos,xpos,curvaturepos,resultpos,sumcnt,sumpos;
  uint64_t zposbase,tposbase,xposbase,curvatureposbase;
  float sum=0.0f;
  int loopdone=FALSE;

  float vval,cval;
  uint64_t vidx,vidx2,cidx,cidx2;
  uint64_t point_vidx[4],point_cidx[4];
  float vidxval,cidxval;
  float integralevalpt, integral_dintegranddvevalpt,integral_dintegranddcevalpt;
  float vals[4],weights[4],totalweight;
  float est,scalarresult;
  //float r_conductivityscaled_sq_ov_z_sq,coeffs,coeff2,one_plus_w;
  float curvcoeff;
  int pointcnt;
  int64_t axiscnt2,axispos;

  resultpos=itercnt;
  
  zposbase=0;
  tposbase=0;
  xposbase=0;
  curvatureposbase=0;

  //fprintf(stderr,"itercnt=%d\n",(int)itercnt);

  for (axiscnt2=0;axiscnt2 < ndim;axiscnt2++) {
    
    if (!axissumflag[axiscnt2]) {
      /* not summing over this axis */
      axispos = resultpos/resultstrides[axiscnt2];
      resultpos -= axispos*resultstrides[axiscnt2];
      
      if (zvecshape[axiscnt2] > 1) {
	/* not broadcasting z over this axis */
	zposbase += axispos*zvecstrides[axiscnt2];
      }
      if (xvecshape[axiscnt2] > 1) {
	/* not broadcasting r over this axis */
	xposbase += axispos*xvecstrides[axiscnt2];
      }
      if (tvecshape[axiscnt2] > 1) {
	/* not broadcasting r over this axis */
	tposbase += axispos*tvecstrides[axiscnt2];
      }
      if (curvature_flag && curvaturevecshape[axiscnt2] > 1) {
	/* not broadcasting curvature over this axis */
	curvatureposbase += axispos*curvaturevecstrides[axiscnt2];
      }
      
    }
    
    
  }

  for (sumcnt=0;!loopdone;sumcnt++) {
    //fprintf(stderr,"sumcnt=%d\n",(int)sumcnt);

    zpos=zposbase;
    tpos=tposbase;
    xpos=xposbase;
    curvaturepos=curvatureposbase;
    sumpos=sumcnt;
    
    for (axiscnt2=0;axiscnt2 < ndim;axiscnt2++) {
      if (axissumflag[axiscnt2]) {
	axispos = sumpos/sumstrides[axiscnt2];
	sumpos -= axispos*sumstrides[axiscnt2];
      
	if (axispos >= shape[axiscnt2]) {
	  loopdone=TRUE;
	  break;
	}
	//fprintf(stderr,"iterationstrides[%lu]=%lu\n",axiscnt2,iterationstrides[axiscnt2]);
	/* summing over this axis */
	
	if (zvecshape[axiscnt2] > 1) {
	  /* not broadcasting z over this axis */
	  zpos += axispos*zvecstrides[axiscnt2];
	}
	if (xvecshape[axiscnt2] > 1) {
	  /* not broadcasting r over this axis */
	  xpos += axispos*xvecstrides[axiscnt2];
	}
	if (tvecshape[axiscnt2] > 1) {
	  /* not broadcasting r over this axis */
	  tpos += axispos*tvecstrides[axiscnt2];
	}
	if (curvature_flag && curvaturevecshape[axiscnt2] > 1) {
	  /* not broadcasting curvature over this axis */
	  curvaturepos += axispos*curvaturevecstrides[axiscnt2];
	}
      }
    }
    if (sumpos > 0 || loopdone) break;
    //fprintf(stderr,"zpos=%d\n",(int)zpos);
    //fprintf(stderr,"xpos=%d\n",(int)xpos);
    //fprintf(stderr,"tpos=%d\n",(int)tpos);

    
    assert(tvec[tpos] > 0);
    
    vval=(4*alphaz*tvec[tpos])/pow(zvec[zpos],2.0f);

    // Flat case:

    if (!curvature_flag) {
      cval=fabs(sqrt(pow(xvec[xpos],2.0f)*(alphaz/alphaxy) + pow(yval,2.0f)*(alphaz/alphaxy) + pow(zvec[zpos],2.0f))/zvec[zpos]);
      //cval=fabs(rconductivityscaledvec[xpos]/zvec[zpos]);

      curvcoeff = 1.0f;
      
    } else {
      // curved case:
      float w_root_v = (curvaturevec[curvaturepos]*zvec[zpos]*sqrt(M_PI)/8.0f)*sqrt(vval);
      // bounds on w_root_v (empirical)
      if (w_root_v < -0.6f) {
	w_root_v = -0.6f;
      }
      if (w_root_v > 1.0f) {
	w_root_v = 1.0f;
      }
      
      curvcoeff = (1.0f/(1.0f + 0.8f*w_root_v));
      if (curvaturevec[curvaturepos] >= 0) {
	// Concave
	// cval from greenfcn_doc.pdf eq. 38
	float theta = curvaturevec[curvaturepos]*xvec[xpos];
	if (fabs(theta) > M_PI/2.0f) {
	  theta=M_PI/2.0f;
	}
	float deceleration = pow(1.0f+(1.0f/12.0f)*pow(theta,2.0f),2.0f*(1.0f - xvec[xpos]/(fabs(xvec[xpos])*fabs(theta/2.0f + pow(theta,3.0f)/24.0f))));
	if (deceleration < 1.0f) {
	  deceleration = 1.0f;
	}

	// Value for cedilla
	cval=sqrt(pow(xvec[xpos],2.0f)*(1.0f+zvec[zpos]*curvaturevec[curvaturepos])*deceleration*(alphaz/alphaxy) + pow(yval,2.0f)*(alphaz/alphaxy) + pow(zvec[zpos],2.0f))/zvec[zpos];

	
      } else {
	// Concave
	// cval from greenfcn_doc.pdf eq. 38
	cval=sqrt( pow((1.0f/fabs(curvaturevec[curvaturepos]))-zvec[zpos],2.0f)*(1.0f + fabs(curvaturevec[curvaturepos])*zvec[zpos]/(1.0f - fabs(curvaturevec[curvaturepos])*zvec[zpos]))*4.0f*inner_sin_sq(fabs(curvaturevec[curvaturepos])*xvec[xpos]/2.0f)*(alphaz/alphaxy) + pow(yval,2.0f)*(alphaz/alphaxy) + pow(zvec[zpos],2.0f))/zvec[zpos];

	
      }

      
      /*  Old code: 
      coeffs=curvaturevec[zpos]*sqrt(alphaz*tvec[tpos]);
      coeff2=curvaturevec[zpos]*zvec[zpos];

      one_plus_w=1.0+coeff2; // concave case

      if (curvaturevec[zpos] < 0) {
	// Need to bound 1+w >= 0  
	// where w is coeff2/(1+coeff2), coeff2 negative
	// and (1+coeff2) should not be negative or zero

	// in the bound, we force 1+w to 0

	if (1.0f+coeff2 <= 0.0) {
	  one_plus_w=0.0; // lateral flow is instantaneous
	} else {
	  // constraint is that 1+w positive, i.e. 
	  //   w >= -1
	  //  coeff2/(1+coeff2) >= -1,  where 1+coeff2 > 0
	  //   coeff2 >= -(1+coeff2)
	  //   2*coeff2 >= -1
	  //  coeff2 >= -0.5
	  if (coeff2 >= -0.5) {
	    one_plus_w = 1+coeff2/(1+coeff2);  // convex case
	  } else {
	    one_plus_w=0.0;
	  }
	}
      }
      
      r_conductivityscaled_sq_ov_z_sq=pow(rconductivityscaledvec[rpos]/zvec[zpos],2.0f);
      cval=sqrt((r_conductivityscaled_sq_ov_z_sq-1.0f)*one_plus_w + 1);
      

      // Leading factor scaling according to curvature empirically fitted coefficients
      coeff *= 1.0/(1.0 + tc*coeffs + dc*coeff2 + tc2*pow(coeffs,2.0f) + dc2*pow(coeff2,2.0f) + tcdc*coeffs*coeffs + tc3*pow(coeffs,3.0f) + tc2dc*pow(coeffs,2.0f)*coeff2);
      */
    }
    
    
    
    // print("%f, %f, %f" % (log10(vval),log10v0,dlog10v))
    
    vidx=(int64_t)((log10(vval)-log10v0)/dlog10v);
    // print("vidx=%d; nvrange=%d" % (vidx,nvrange))
    assert(vidx >= 0 && vidx+1 < nvrange);
    vidx2=vidx+1;
    
    cidx=(int64_t)((log10(cval)-log10c0)/dlog10c);
    // print("cidx=%d; ncrange=%d" % (cidx,ncrange))
    assert(cidx >= 0 && cidx+1 < ncrange); 
    cidx2=cidx+1;
    
    point_vidx[0]=vidx;
    point_cidx[0]=cidx;
    
    point_vidx[1]=vidx;
    point_cidx[1]=cidx2;
    
    point_vidx[2]=vidx2;
    point_cidx[2]=cidx;
    
    point_vidx[3]=vidx2;
    point_cidx[3]=cidx2;
    
    totalweight=0.0f;
    
    for (pointcnt=0;pointcnt < 4;pointcnt++) {
      vidxval=vrange[point_vidx[pointcnt]];
      cidxval=crange[point_cidx[pointcnt]];
      
      integralevalpt=integraleval[point_vidx[pointcnt]*ncrange + point_cidx[pointcnt]];
      integral_dintegranddvevalpt=integral_dintegranddveval[point_vidx[pointcnt]*ncrange + point_cidx[pointcnt]];
      integral_dintegranddcevalpt=integral_dintegranddceval[point_vidx[pointcnt]*ncrange + point_cidx[pointcnt]];
      vals[pointcnt] = integralevalpt + (vval-vidxval)*integral_dintegranddvevalpt +(cval-cidxval)*integral_dintegranddcevalpt;
      weights[pointcnt]=sqrt((float)(1.0f/(0.001f+(vval-vidxval)*(vval-vidxval) + (cval-cidxval)*(cval-cidxval))));
      
      totalweight+=weights[pointcnt];
      
    }
    
    est=0.0f;
    for (pointcnt=0;pointcnt < 4;pointcnt++) {
      est+=vals[pointcnt]*weights[pointcnt]/totalweight;
    }
    
    // Limit according to nonnegative and upper bound in greensfcn_doc.tex
    if (est < 0.0f) {
      // print("Warning: Integral gave inaccurate calculation of %g at v=%g, c=%g; lower bound of 0 used instead" % (est,vval,cval),file=sys.stderr)
      est=0.0f;
    } else if (est > 0.185f*exp((float)(-(pow(cval,2.0f)-1.0f)/vval))) {
#ifndef __OPENCL_VERSION__
      fprintf(stderr,"Warning: Integral gave inaccurate calculation of %g at v=%g,c=%g; upper bound of %g used instead\n",est,vval,cval,0.185f*exp((float)(-(pow(cval,2.0f)-1.0f)/vval)));
#endif /* __OPENCL_VERSION__ */
      est= 0.185f*exp((float)(-(pow(cval,2.0f)-1.0f)/vval));
    }
    
    scalarresult=coeff*curvcoeff*est/pow(zvec[zpos],3.0f);
    sum+=scalarresult;
    
  }
  
  
  //#ifdef USE_OPENMP
  //#pragma omp atomic // or #pragma omp atomic update
  //#endif /* USE_OPENMP */
  result[itercnt]=sum; //# This assignment and increment must be atomic
  
}

void greensconvolution_integrate_anisotropic_c(
					       OPENCL_GLOBAL const float *vrange,uint64_t nvrange, //number of rows in integraleval
					       OPENCL_GLOBAL const float *crange,uint64_t ncrange, // number of cols in integraleval
					       OPENCL_GLOBAL const float *integraleval,
					       OPENCL_GLOBAL const float *integral_dintegranddveval, // same size as integraleval
					       OPENCL_GLOBAL const float *integral_dintegranddceval, // same size as integraleval
					       OPENCL_GLOBAL const float *zvec, OPENCL_GLOBAL const uint64_t *zvecshape,OPENCL_GLOBAL const uint64_t *zvecstrides,
					       OPENCL_GLOBAL const float *xvec, OPENCL_GLOBAL const uint64_t *xvecshape,OPENCL_GLOBAL const uint64_t *xvecstrides,
					       OPENCL_GLOBAL const float *tvec, OPENCL_GLOBAL const uint64_t *tvecshape,OPENCL_GLOBAL const uint64_t *tvecstrides,
					       float yval,
					       OPENCL_GLOBAL const float *curvaturevec, OPENCL_GLOBAL const uint64_t *curvaturevecshape, OPENCL_GLOBAL const uint64_t *curvaturevecstrides,

					       OPENCL_GLOBAL const uint64_t *sumstrides, OPENCL_GLOBAL const uint64_t *shape,
					       float alphaz,float alphaxy,int8_t curvature_flag,
					       OPENCL_GLOBAL float *result,OPENCL_GLOBAL const uint64_t *resultstrides,
					       float coeff,
					       OPENCL_GLOBAL const uint64_t *axissumflag,
					       uint64_t ndim)  
{
  // NOTE: This is supposed to be the same code as in greensconvolution_fast.pyx/greensconvolution_integrate_anisotropic_py
  float log10v0,log10c0,dlog10v,dlog10c;
  uint64_t iterlen;
  uint64_t itercnt;
  uint64_t axiscnt;

  iterlen=1;

  for (axiscnt=0;axiscnt < ndim;axiscnt++) {
    if (!axissumflag[axiscnt]) {
      iterlen *= shape[axiscnt];
    }
  }
  

  assert(nvrange > 0);
  assert(ncrange > 0);

  log10v0=log10(vrange[0]);
  log10c0=log10(crange[0]);
  dlog10v=log10(vrange[1])-log10(vrange[0]);
  dlog10c=log10(crange[1])-log10(crange[0]);


#ifdef USE_OPENMP
#pragma omp parallel for shared(tvec,zvec,xvec,result,vrange,crange,integraleval,integral_dintegranddveval,integral_dintegranddceval,stderr,alphaz,alphaxy,curvature_flag,log10v0,log10c0,dlog10v,dlog10c,nvrange,ncrange,coeff,ndim,resultstrides,tvecshape,xvecshape,zvecshape,tvecstrides,xvecstrides,zvecstrides,yval,curvaturevec,curvaturevecshape,curvaturevecstrides,sumstrides,shape,iterlen,axissumflag) default(none) private(itercnt)
#endif /* USE_OPENMP */
  for (itercnt=0; itercnt < iterlen; itercnt++) {
    greensconvolution_integrate_anisotropic_c_one(itercnt,
						  vrange,nvrange, //number of rows in integraleval
						  crange,ncrange, // number of cols in integraleval
						  integraleval,
						  integral_dintegranddveval, // same size as integraleval
						  integral_dintegranddceval, // same size as integraleval
						  zvec, zvecshape,zvecstrides,
						  xvec, xvecshape,xvecstrides,
						  tvec, tvecshape,tvecstrides,
						  yval,
						  curvaturevec,curvaturevecshape,curvaturevecstrides,
						  sumstrides, shape,		       	  
						  result,resultstrides,
						  coeff,
						  axissumflag,
						  ndim,
						  log10v0,log10c0,dlog10v,dlog10c,alphaz,alphaxy,curvature_flag);
    
    
  }

  

  
}

#ifdef __OPENCL_VERSION__
OPENCL_KERNEL
void greensconvolution_integrate_anisotropic_c_opencl(OPENCL_GLOBAL const float *vrange,uint64_t nvrange, //number of rows in integraleval
						      OPENCL_GLOBAL const float *crange,uint64_t ncrange, // number of cols in integraleval
						      OPENCL_GLOBAL const float *integraleval,
						      OPENCL_GLOBAL const float *integral_dintegranddveval, // same size as integraleval
						      OPENCL_GLOBAL const float *integral_dintegranddceval, // same size as integraleval
						      OPENCL_GLOBAL const float *zvec, OPENCL_GLOBAL const uint64_t *zvecshape,OPENCL_GLOBAL const uint64_t *zvecstrides,
						      OPENCL_GLOBAL const float *xvec, OPENCL_GLOBAL const uint64_t *xvecshape,OPENCL_GLOBAL const uint64_t *xvecstrides,
						      OPENCL_GLOBAL const float *tvec, OPENCL_GLOBAL const uint64_t *tvecshape,OPENCL_GLOBAL const uint64_t *tvecstrides,
						      float yval,
						      OPENCL_GLOBAL const uint64_t *sumstrides,OPENCL_GLOBAL const uint64_t *shape,
						      float alphaz,float alphaxy,
						      OPENCL_GLOBAL float *result,OPENCL_GLOBAL const uint64_t *resultstrides,
						      float coeff,
						      OPENCL_GLOBAL const uint64_t *axissumflag,
						      uint64_t ndim)  
{
  uint64_t itercnt=get_global_id(0);


  float log10v0,log10c0,dlog10v,dlog10c;

  //alphaz=kz*1.0f/(rho*cp);
  
  //alphaxyz=pow((kx*ky*kz),(1.0f/3.0f))/(rho*cp);

  //coeff*=2.0f*pow(alphaz,(3.0f/2.0f))/((rho*cp*M_PI*M_PI)*pow(alphaxyz,(3.0f/2.0f)));

  log10v0=log10(vrange[0]);
  log10c0=log10(crange[0]);
  dlog10v=log10(vrange[1])-log10(vrange[0]);
  dlog10c=log10(crange[1])-log10(crange[0]);

  
  greensconvolution_integrate_anisotropic_c_one(itercnt,
						vrange,nvrange, //number of rows in integraleval
						crange,ncrange, // number of cols in integraleval
						integraleval,
						integral_dintegranddveval, // same size as integraleval
						integral_dintegranddceval, // same size as integraleval
						zvec, zvecshape,zvecstrides,
						xvec, xvecshape,xvecstrides,
						tvec, tvecshape,tvecstrides,
						yval,
						NULL,NULL,NULL,
						sumstrides, shape,
						
						result,resultstrides,
						coeff,
						axissumflag,
						ndim,
						log10v0,log10c0,dlog10v,dlog10c,alphaz,alphaxy,FALSE);
  
  
  
}


OPENCL_KERNEL
void greensconvolution_integrate_anisotropic_curved_c_opencl(OPENCL_GLOBAL const float *vrange,uint64_t nvrange, //number of rows in integraleval
								OPENCL_GLOBAL const float *crange,uint64_t ncrange, // number of cols in integraleval
								OPENCL_GLOBAL const float *integraleval,
								OPENCL_GLOBAL const float *integral_dintegranddveval, // same size as integraleval
								OPENCL_GLOBAL const float *integral_dintegranddceval, // same size as integraleval
								OPENCL_GLOBAL const float *zvec, OPENCL_GLOBAL const uint64_t *zvecshape,OPENCL_GLOBAL const uint64_t *zvecstrides,
								OPENCL_GLOBAL const float *xvec, OPENCL_GLOBAL const uint64_t *xvecshape,OPENCL_GLOBAL const uint64_t *xvecstrides,
								OPENCL_GLOBAL const float *tvec, OPENCL_GLOBAL const uint64_t *tvecshape,OPENCL_GLOBAL const uint64_t *tvecstrides,
							     float yval,
								OPENCL_GLOBAL const float *curvaturevec, OPENCL_GLOBAL const uint64_t *curvaturevecshape, OPENCL_GLOBAL const uint64_t *curvaturevecstrides,
								OPENCL_GLOBAL const uint64_t *sumstrides,OPENCL_GLOBAL const uint64_t *shape,
							     float alphaz,float alphaxy,
								OPENCL_GLOBAL float *result,OPENCL_GLOBAL const uint64_t *resultstrides,
								float coeff,
								OPENCL_GLOBAL const uint64_t *axissumflag,
								uint64_t ndim)  
{
  uint64_t itercnt=get_global_id(0);


  float log10v0,log10c0,dlog10v,dlog10c;

  //alphaz=kz*1.0f/(rho*cp);
  
  //alphaxyz=pow((kx*ky*kz),(1.0f/3.0f))/(rho*cp);

  //coeff*=2.0f*pow(alphaz,(3.0f/2.0f))/((rho*cp*M_PI*M_PI)*pow(alphaxyz,(3.0f/2.0f)));

  log10v0=log10(vrange[0]);
  log10c0=log10(crange[0]);
  dlog10v=log10(vrange[1])-log10(vrange[0]);
  dlog10c=log10(crange[1])-log10(crange[0]);

  
  greensconvolution_integrate_anisotropic_c_one(itercnt,
						vrange,nvrange, //number of rows in integraleval
						crange,ncrange, // number of cols in integraleval
						integraleval,
						integral_dintegranddveval, // same size as integraleval
						integral_dintegranddceval, // same size as integraleval
						zvec, zvecshape,zvecstrides,
						xvec, xvecshape,xvecstrides,
						tvec, tvecshape,tvecstrides,
						yval,
						curvaturevec,curvaturevecshape,curvaturevecstrides,
						sumstrides, shape,
						
						result,resultstrides,
						coeff,
						axissumflag,
						ndim,
						log10v0,log10c0,dlog10v,dlog10c,alphaz,alphaxy,TRUE);
  
  
  
}


#endif /* __OPENCL_VERSION__ */
