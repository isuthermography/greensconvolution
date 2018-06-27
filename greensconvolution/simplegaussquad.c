/* OpenCL kernel for simple Gaussian quadrature integration */
#ifndef TRUE
#define TRUE (!0)
#endif

#ifndef FALSE
#define FALSE (0)
#endif

typedef unsigned long uint64_t;
typedef long int64_t;

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif


#define OFFSET 0.5f /* (epsilon + 1.0-epsilon)/2.0 */
//#define DEGREE 300 // filled in when loaded
//#define NUM_REPEATS 1000 /***!!! FOR PERFORMANCE TESTING ONLY... REDUCE THIS  DOWN TO 1 OTHERWISE! **/

float integrand(float x, float v, float a) 
{
  return pow(x,-1.5f)*pow(1.0f-x,-1.5f)*exp(-(1.0f+a*x)/(v*x*(1.0f-x)));
}


__kernel void simplegaussquad_opencl(__global const float *gauss_points,
				     __global const float *gauss_weights, 
				     __global const float *zvec, __global const uint64_t *zvecshape,__global const uint64_t *zvecstrides,
				     __global const float *xvec, __global const uint64_t *xvecshape,__global const uint64_t *xvecstrides,
				     __global const float *tvec, __global const uint64_t *tvecshape,__global const uint64_t *tvecstrides,
				     float yval,
				     __global const uint64_t *sumstrides, __global const uint64_t *shape,
				     float alphaz,
				     __global float *result,__global const uint64_t *resultstrides,
				     float coeff,
				     __global const uint64_t *axissumflag,
				     uint64_t ndim)  
{
  uint64_t itercnt=get_global_id(0);

  uint64_t zpos,tpos,xpos,resultpos,sumcnt,sumpos;
  uint64_t zposbase,tposbase,xposbase;
  float sum=0.0;
  int64_t axiscnt2,axispos;
  int loopdone=FALSE;

  int cnt;



  float epsilon,leadingcoeff_or_slope;
  float my_a,my_v,my_c,accum;
  



  resultpos=itercnt;

  zposbase=0;
  tposbase=0;
  xposbase=0;
  

  for (axiscnt2=0;axiscnt2 < ndim;axiscnt2++) {
    
    if (!axissumflag[axiscnt2]) {
      /* not summing over this axis */
      axispos = resultpos/resultstrides[axiscnt2];
      resultpos -= axispos*resultstrides[axiscnt2];
      
      if (zvecshape[axiscnt2] > 1) {
	/* not broadcasting z over this axis */
	zposbase += axispos*zvecstrides[axiscnt2];
      }
      if (rvecshape[axiscnt2] > 1) {
	/* not broadcasting r over this axis */
	xposbase += axispos*xvecstrides[axiscnt2];
      }
      if (tvecshape[axiscnt2] > 1) {
	/* not broadcasting r over this axis */
	tposbase += axispos*tvecstrides[axiscnt2];
      }
      
      
    }
    
    
  }

  for (sumcnt=0;!loopdone;sumcnt++) {
    //fprintf(stderr,"sumcnt=%d\n",(int)sumcnt);

    zpos=zposbase;
    tpos=tposbase;
    xpos=xposbase;
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
	
      }
    }
    if (sumpos > 0 || loopdone) break;
    
  
    my_c=fabs(sqrt(pow(xvec[xpos],2.0f)*(alphaz/alphaxy) + pow(yval,2.0f)*(alphaz/alphaxy) + pow(zvec[zpos],2.0f))/zvec[zpos]);
    my_a=my_c*my_c-1.0;
    my_v=(4*alphaz*tvec[tpos])/pow(zvec[zpos],2.0f);
    
  epsilon = 1.0f; 
  if (my_v > epsilon) {
    epsilon=my_v;
  } 
  if (my_a > epsilon) {
    epsilon=my_a;
  }
  epsilon = .05f/epsilon; 

  leadingcoeff_or_slope = (1.0f-2.0f*epsilon)/2.0f;

  /* vals = (1.0/v**2.0) * leadingcoefficient * np.tensordot(weights,np.vectorize(integrand)(slope*points+offset,v,a),axes=(0,0)) */

  accum=0.0f;     
  /* Note: On Intel GPU it gets quite a bit faster if degree is a constant rather than a parameter (!) */
  for (cnt=0;cnt < DEGREE;cnt++) {
    accum += gauss_weights[cnt] * integrand(leadingcoeff_or_slope * gauss_points[cnt] + OFFSET,my_v,my_a);
    
  }
  accum *= (coeff/pow(zvec[zpos],3.0f))*leadingcoeff_or_slope/pow(my_v,2.0f);

  sum+=accum;
  }
  result[itercnt]=sum;

}
