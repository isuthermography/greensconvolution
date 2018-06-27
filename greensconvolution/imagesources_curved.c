/* OpenCL kernel for calculating contributions of image sources (curved surfaces) */
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


__kernel void imagesources_curved_opencl(__global const float *rconductivityscaledvec, __global const uint64_t *rvecshape,__global const uint64_t *rvecstrides,
					 __global const float *tvec, __global const uint64_t *tvecshape,__global const uint64_t *tvecstrides,
					 __global const float *avgcurvatures, __global const uint64_t *avgcurvaturesshape, __global const uint64_t *avgcurvaturesstrides,
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
  uint64_t tposbase,rposbase,avgcurvaturesposbase;
  float sum=0.0f,thiscoeff,curvature,Predicted_T,Exponent;
  int64_t axiscnt2,axispos;
  float r2pos;


  // Coefficients, from eval_curved_laminate.py 10/25/16 02:32pm
  // THESE PROBABLY NEED TO BE RECALCULATED FOR THE SIMPLIFIED LAMINATE
  // OR DETERMINED FROM THEORY!!!
  //const float tc=0.40211414;
  //const float dc=-0.00423593;
  //const float tc2= -0.03041781;
  //const float dc2=0.0027296;
  //const float tcdc=-0.05028247;
  //const float tc3=-0.00087789;
  //const float  tc2dc=0.00723961;


  resultpos=itercnt;

  tposbase=0;
  rposbase=0;
  avgcurvaturesposbase=0;


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

    if (avgcurvaturesshape[axiscnt2] > 1) {
      /* not broadcasting r over this axis */
      avgcurvaturesposbase += axispos*avgcurvaturesstrides[axiscnt2];
    }

  }
  
  /* Loop over images (z positions)  */


  /* 
     For curved surfaces, this routine provides the high order images. 
     In a flat specimen, for a source at depth z, these would be located at
     +/- 4z, 6z 8z 10z etc. (The "original" at 2z is provided instead by the base
     curved greensconvolution calculation)

     We get as our parameter, only the positive 
     z coordinates of these images, i.e. 4z. 6z, 8z, etc.
     
     In the flat case, this is all quite simple.. we are given 
     a leading coefficient (which includes the factor of 2 representing
     the image sources at negative z, plus the leading constants in 
     the Green's function) and we just evaluate the 3D Green's function 
     formula. 

     In this case, our surface is curved. 

     We use the Green's function approximation 
     to approximate heatflow from a source in the curved surface. 
     
     In the flat case, you have image sources on both sides of the 
     z=0 free surface, but the formulas are identical so we just 
     calculate it here for the z > 0 case and the factor of 2 externally
     provided takes care of the z < 0 case. 

     In the curved case, these image sources are somewhat different. 
     (and no longer work perfectly to satisfy the B.C!) 
     In particular, they get narrower on the concave side 
     and broader on the convex side. 
     Also, because they get moved significantly toward the 
     center of curvature on the concave side, they could get
     very small or even drop to zero size (or even negative size, 
     mathmatically). Obviously this is problematic. 
     
     The main effect of the source location approaching the center
     of curvature is that the time required for lateral heat flow 
     no longer is significant -- because all points at a given 
     radius heat simultaneously. So points very far away would show 
     heating (at a late time, etc.). In any case this doesn't seem
     right. 

     So what we do is only represent the image on the convex side, where
     it can be as big as necessary without 
     problems. And we let the caller still provide that leading factor
     of 2 representing the fact that we represent both concave side
     and convex side image sources by doubling the magnitude 
     of the convex side source. 

     We use only the pieces of the approximate curved surface Green's 
     function that are likely to be significant for this
     scenario: 
       * The ExtraVolumeFactor corresponding to the original buried 
         source
       * The appropriate depth with the z axis diffusivity
       * The surface lateral distance with the x axis diffusivity
         (presumed to average shortcuts on one side and longer 
         distances on the other
       * no ther corrections.  

     Conveniently all this reduces to the flat case when the 
     curvature goes to zero. 

     Regardless of the sign of the curvature, we will always
     use the concave formula (the concave surface is the free
     "z=0" surface) because that puts the image source on the
     convex side and won't cause problems with the source 
     being beyond the center of curvature.  
 
     We will use the 'extravolumefactor' according to the 
     actual sign of the curvature, though. 
     
  */
  
  
  /* leading coefficient of */
  // Formula from curved_laminate_simplified.py for a buried point source :
  // source_energy*(2.0/(rho*c*(4*np.pi)**(3.0/2.0)*np.sqrt(alphaz)*alphaxy))
  // Simplifies down to:
  //
  // Sourcevecs.py already multiplies by reflector_widthy*reflector_widthx (integral over source area)*2.0
  // for unit source energy

  // greensconvolution_fast.pyx multiplies by (2.0/(rho*c)) / (4 pi alphaxyz)^(3/2)
  // so we don't have to include those factors
  //
  // Sourcevecs+greensconvolution_fast, neglecting the area integral, provide a leading factor (coeff) of:
  //     source_energy*2.0 *  (2.0/rho c) / ( (4 pi alphaxyz)^(3/2)  )
  //          since alpha = k/(rho c), k = alpha rho c 
  // becomes   source_energy*2.0 *  (2.0/rho c) / ( (4 pi)^(3/2) * sqrt(alphaz)*alphaxy )  
  // i.e. exactly twice the formula from curved_laminate_simplified.py
  // with the "twice" accounting for the "-z" image source copy.
  //
  // So our leading coefficient is just "coeff"

  // Next coefficient from curved_laminate_simplified.py:
  //  1.0/Pred_t**(3.0/2.0)
  
  thiscoeff=coeff/pow(tvec[tposbase],(float)(3.0f/2.0f));


  //curvature=avgcurvatures[avgcurvaturesposbase];
  //if (curvature < 0.0) {
  //  curvature=-curvature; // positive curvature means concave free surface, image sources on convex side. Force curvature positive to properly place the image sources. 
  //}

  
  float ExtraVolumeFactor=0.25f*avgcurvatures[avgcurvaturesposbase]*sqrt(M_PI*alphaz*tvec[tposbase]);
  // Bounds on ExtraVolumeFactor
  if (ExtraVolumeFactor > 1.0f) {
    ExtraVolumeFactor=1.0f;
  }
  if (ExtraVolumeFactor < -0.6f) {
    ExtraVolumeFactor=-0.6f;
  }
  

  
  // All of the above was independent of what z position we are at. 
  
  for (imagenum=0;imagenum < nimages;imagenum++) {
    //barrier(CLK_LOCAL_MEM_FENCE);
    
    
    Predicted_T = thiscoeff/(1.0f+ExtraVolumeFactor); // last factor from above loop
    //printf("WithEVF: %f\n",Predicted_T);
    // Next factor: np.exp(-source_depth**2.0/(4.0*alphaz*Pred_t))
    // (put into Exponent) 
    Exponent = -pow(image_source_zposns[imagenum],2.0f)/(4.0f*alphaz*tvec[tposbase]) - pow(rconductivityscaledvec[rposbase],2.0f)/(4.0f*alphaz*tvec[tposbase]);
    // factor in the exponent
    Predicted_T *= exp(Exponent);
    //printf("WithEXP: %f\n",Predicted_T);

    // Add effect of this image source to our sum
    
    sum+=Predicted_T;
    
  }
  
  result[itercnt]=sum;

}
