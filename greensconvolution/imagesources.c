/* OpenCL kernel for calculating contributions of image sources (flat surfaces) */
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


__kernel void imagesources_opencl(__global const float *rconductivityscaledvec, __global const uint64_t *rvecshape,__global const uint64_t *rvecstrides,
				  __global const float *tvec, __global const uint64_t *tvecshape,__global const uint64_t *tvecstrides,
				  __global const uint64_t *shape,
				  float alphaz,
				  __global float *result,__global const uint64_t *resultstrides,
				  float coeff,
				  uint64_t ndim,
				  __global const float *image_source_zposns,
				  uint64_t nimages)  
{
  uint64_t itercnt=get_global_id(0);

  uint64_t imagenum;
  
  uint64_t resultpos;
  uint64_t tposbase,rposbase;
  float sum=0.0,thiscoeff,expcoeff;
  int64_t axiscnt2,axispos;
  float r2pos;




  resultpos=itercnt;

  tposbase=0;
  rposbase=0;
  


  for (axiscnt2=0;axiscnt2 < ndim;axiscnt2++) { 
    
    /* not summing over this axis */
    axispos = resultpos/resultstrides[axiscnt2];
    resultpos -= axispos*resultstrides[axiscnt2];
      
    if (rvecshape[axiscnt2] > 1) {
      /* not broadcasting r over this axis */
      rposbase += axispos*rvecstrides[axiscnt2];
    }
    if (tvecshape[axiscnt2] > 1) {
      /* not broadcasting r over this axis */
      tposbase += axispos*tvecstrides[axiscnt2];
    }
      
  }
  
  /* Loop over images (z positions)  */

  /* leading coefficient */  
  thiscoeff=coeff/pow(tvec[tposbase],(float)(3.0/2.0));
  expcoeff=4.0*alphaz;


  for (imagenum=0;imagenum < nimages;imagenum++) {
    //barrier(CLK_LOCAL_MEM_FENCE);
    
    r2pos=pow(rconductivityscaledvec[rposbase],(float)2.0) + pow(image_source_zposns[imagenum],(float)2.0);

    sum+=exp(-r2pos/(expcoeff*tvec[tposbase]));
    
  }
  
  result[itercnt]=sum*thiscoeff;

}
