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


// deceleration from concave analysis
static inline float deceleration(float theta,float x,float depth)
{
  if (fabs(theta) > M_PI/2.0f) {
    theta=M_PI/2.0f;
  }
  
  if (depth < fabs(x)*fabs(theta/2.0f + pow(theta,3.0f)/24.0f)) {        
    return pow(1.0f/(1.0f+(1.0f/12.0f)*pow(theta,2.0f)),-2.0f * (1.0f - depth/(fabs(x)*fabs(theta/2.0f + pow(theta,3.0f)/24.0f))));
  }
  return 1.0f;
}

// inner_sin_sq from convex analysis
static inline float inner_sin_sq(float x)
{
  if (fabs(x) <= M_PI/4.0f) {
    return pow(sin(x),2.0f);
  }
  return 0.25f+pow(fabs(x)-((float)M_PI)/4.0f+0.5f,2.0f);
  
}


__kernel void greensfcn_curved_opencl(
					 __global const float *linelength, __global const uint64_t *linelengthshape,__global const uint64_t *linelengthstrides,
					 __global const float *tvec, __global const uint64_t *tvecshape,__global const uint64_t *tvecstrides,
					 __global const float *source_intensity, __global const uint64_t *source_intensityshape,__global const uint64_t *source_intensitystrides,
					 __global const float *depth, __global const uint64_t *depthshape,__global const uint64_t *depthstrides,
					 __global const float *theta, __global const uint64_t *thetashape,__global const uint64_t *thetastrides, /* theta=0 -> line parallel to x, theta=pi/2 -> line parallel to y */
					 __global const float *avgcurvatures, __global const uint64_t *avgcurvaturesshape, __global const uint64_t *avgcurvaturesstrides,
					 __global const float *avgcrosscurvatures, __global const uint64_t *avgcrosscurvaturesshape, __global const uint64_t *avgcrosscurvaturesstrides,
					 __global const float *iop_dy, __global const uint64_t *iop_dy_shape, __global const uint64_t *iop_dy_strides,
					 __global const float *iop_dx, __global const uint64_t *iop_dx_shape, __global const uint64_t *iop_dx_strides,
					 __global const uint64_t *shape,
					 __global const uint64_t *axissumflag,
					 __global const uint64_t *sumstrides,
					 float kz,float ky,float kx,float rho,float cp,
					 __global float *result,__global const uint64_t *resultshape,__global const uint64_t *resultstrides,
					 uint64_t ndim)  
/* iop is integrate-over-pixel, allowing us to evaluate the integral over a small pixel rather than evaluate it at a precise point 
   set the iop_ parameters to NULL to disable or iop_dx or _dy to 0.0 to disable at a point */
{
  uint64_t itercnt=get_global_id(0);

  uint64_t resultpos;
  uint64_t pos;

  uint64_t tposbase,linelengthposbase,avgcurvaturesposbase,avgcrosscurvaturesposbase,iop_dy_posbase,iop_dx_posbase,source_intensityposbase,thetaposbase,depthposbase;
  uint64_t tpos,linelengthpos,avgcurvaturespos,avgcrosscurvaturespos,iop_dy_pos,iop_dx_pos,source_intensitypos,depthpos,thetapos;
  uint64_t sumcnt,sumpos,loopdone=FALSE;
  uint64_t axiscnt2,axispos;

  float sum=0.0f;
  float Predicted_T;
  float ExtraVolumeFactor,LengthFactor;
  float alphaz,alphax,alphay;
  float iop_dparallel=0.0f,iop_dcross=0.0f;

  //iop_dx=(void*)0;
  //iop_dy=(void*)0;
  
  resultpos=itercnt;

  tposbase=0;
  linelengthposbase=0;
  source_intensityposbase=0;
  depthposbase=0;
  thetaposbase=0;
  avgcurvaturesposbase=0;
  avgcrosscurvaturesposbase=0;
  iop_dy_posbase=0;
  iop_dx_posbase=0;

  alphaz=kz/(rho*cp);
  alphay=ky/(rho*cp);
  alphax=kx/(rho*cp);
  

  for (axiscnt2=0;axiscnt2 < ndim;axiscnt2++) { 
    if (!axissumflag[axiscnt2]) {
      /* not summing over this axis */
      axispos = resultpos/resultstrides[axiscnt2];
      resultpos -= axispos*resultstrides[axiscnt2];
    
      if (linelengthshape[axiscnt2] > 1) {
	/* not broadcasting r over this axis */
	linelengthposbase += axispos*linelengthstrides[axiscnt2];
      }
      if (tvecshape[axiscnt2] > 1) {
	/* not broadcasting r over this axis */
	tposbase += axispos*tvecstrides[axiscnt2];
      }
      if (source_intensityshape[axiscnt2] > 1) {
	/* not broadcasting source_intensity over this axis */
	source_intensityposbase += axispos*source_intensitystrides[axiscnt2];
      }
      if (depthshape[axiscnt2] > 1) {
	/* not broadcasting source_intensity over this axis */
	depthposbase += axispos*depthstrides[axiscnt2];
      }
      if (thetashape[axiscnt2] > 1) {
	/* not broadcasting source_intensity over this axis */
	thetaposbase += axispos*thetastrides[axiscnt2];
      }

      if (avgcurvaturesshape[axiscnt2] > 1) {
	/* not broadcasting r over this axis */
	avgcurvaturesposbase += axispos*avgcurvaturesstrides[axiscnt2];
      }

      if (avgcrosscurvaturesshape[axiscnt2] > 1) {
	/* not broadcasting r over this axis */
	avgcrosscurvaturesposbase += axispos*avgcrosscurvaturesstrides[axiscnt2];
      }
      if (iop_dy && iop_dy_shape[axiscnt2] > 1) {
	/* not broadcasting r over this axis */
	iop_dy_posbase += axispos*iop_dy_strides[axiscnt2];
      }

      if (iop_dx && iop_dx_shape[axiscnt2] > 1) {
	/* not broadcasting r over this axis */
	iop_dx_posbase += axispos*iop_dx_strides[axiscnt2];
      }
    }

  }
  

  for (sumcnt=0;!loopdone;sumcnt++) {
    linelengthpos=linelengthposbase;
    tpos=tposbase;
    source_intensitypos=source_intensityposbase;
    depthpos=depthposbase;
    thetapos=thetaposbase;
    avgcurvaturespos=avgcurvaturesposbase;
    avgcrosscurvaturespos=avgcrosscurvaturesposbase;
    iop_dy_pos=iop_dy_posbase;
    iop_dx_pos=iop_dx_posbase;

    sumpos=sumcnt;

    for (axiscnt2=0;axiscnt2 < ndim;axiscnt2++) {
      if (axissumflag[axiscnt2]) {
	axispos = sumpos/sumstrides[axiscnt2];
	sumpos -= axispos*sumstrides[axiscnt2];

	if (axispos >= shape[axiscnt2]) {
	  loopdone=TRUE;
	  break;
	}
	
	if (linelengthshape[axiscnt2] > 1) {
	  /* not broadcasting r over this axis */
	  linelengthpos += axispos*linelengthstrides[axiscnt2];
	}
	if (tvecshape[axiscnt2] > 1) {
	  /* not broadcasting t over this axis */
	  tpos += axispos*tvecstrides[axiscnt2];
	}
	if (source_intensityshape[axiscnt2] > 1) {
	  /* not broadcasting source_intensity over this axis */
	  source_intensitypos += axispos*source_intensitystrides[axiscnt2];
	}
	if (depthshape[axiscnt2] > 1) {
	  /* not broadcasting source_intensity over this axis */
	  depthpos += axispos*depthstrides[axiscnt2];
	}
	if (thetashape[axiscnt2] > 1) {
	  /* not broadcasting source_intensity over this axis */
	  thetapos += axispos*thetastrides[axiscnt2];
	}
	
	if (avgcurvaturesshape[axiscnt2] > 1) {
	  /* not broadcasting r over this axis */
	  avgcurvaturespos += axispos*avgcurvaturesstrides[axiscnt2];
	}
	if (avgcrosscurvaturesshape[axiscnt2] > 1) {
	  /* not broadcasting r over this axis */
	  avgcrosscurvaturespos += axispos*avgcrosscurvaturesstrides[axiscnt2];
	}
	if (iop_dy && iop_dy_shape[axiscnt2] > 1) {
	  /* not broadcasting r over this axis */
	  iop_dy_pos += axispos*iop_dy_strides[axiscnt2];
	}
	if (iop_dx && iop_dx_shape[axiscnt2] > 1) {
	  /* not broadcasting r over this axis */
	  iop_dx_pos += axispos*iop_dx_strides[axiscnt2];
	}
      }
    }
    if (sumpos > 0 || loopdone) break;
    
    /* NOTE: To be rigorous we would have to input a full thermal conductivity tensor....
       here we assume alpha = [alphax 0 ; 0 alphay]
       and rotate it by theta: 
            [ cos sin ][ alphax   0    ][ cos -sin ]
            [-sin cos ][   0    alphay ][ sin  cos ]

            [ cos sin ][ alphaxcos  -alphaxsin  ]
            [-sin cos ][ alphaysin   alphaycos ]
	    [ alphaxcos^2+alphaysin^2   -alphaxcossin+alphaycossin
            [-alphaxcossin+alphycossin  alphaxsin^2+alphaycos^2 ]
	    ... and we neglect the off-diagonal elements 
	    
	    Conveniently if alphax and alphay are equal this does nothing (sin^2+cos^2 = 1)
    */

    float alpha_parallel = alphax*pow(cos(theta[thetapos]),2.0f)+alphay*pow(sin(theta[thetapos]),2.0f);
    float alpha_cross = alphay*pow(cos(theta[thetapos]),2.0f)+alphax*pow(sin(theta[thetapos]),2.0f);
    

    /* Now do calculation */
    // see curved_laminate_combined_surfcorr_2d.py and curved_laminate_final_2d.py in heatsim2/demos
    // for heat source on surface of curved half space 
    // theory/verification
    // This is extended to 3D, including effect of cross-curvature
    // (that is presumed to affect the ExtraVolumeFactor but
    // nothing else)
    if (iop_dx && (iop_dx[iop_dx_pos] != 0.0f || iop_dy[iop_dy_pos] != 0.0f)) {
      /* integrate-over-pixel mode */
      

      /* attempt to rotate dx, dy.... not perfect but I don't think there's a perfect answer. 
	 Primary constraint: dx*dy should be conserved */
      iop_dparallel = iop_dx[iop_dx_pos]*pow(cos(theta[thetapos]),2.0f) + iop_dy[iop_dy_pos]*pow(sin(theta[thetapos]),2.0f);

      iop_dcross = iop_dx[iop_dx_pos]*iop_dy[iop_dy_pos]/iop_dparallel;


    } //else {
      //Predicted_T = source_intensity[source_intensitypos]*(2.0/(rho*cp*pow((float)(4.0f*M_PI),(float)3.0f/2.0f)*sqrt(alphaz)*alphax*alphay*pow(tvec[tpos],(float)(3.0f/2.0f))))*exp(-pow(depth[depthpos],2.0f)/(4.0f*alphaz*tvec[tpos]));
    //}
    Predicted_T = source_intensity[source_intensitypos]*(2.0f/((rho*cp)*pow(4.0f*((float)M_PI)*alphaz*tvec[tpos],0.5f)))*exp(-pow(depth[depthpos],2.0f)/(4.0f*alphaz*tvec[tpos]));


    ExtraVolumeFactor = (0.25f)*(avgcurvatures[avgcurvaturespos]+avgcrosscurvatures[avgcrosscurvaturespos])*sqrt(((float)M_PI)*alphaz*tvec[tpos]);
    // ExtraVolumeFactor of 1.0 would halve Predicted_T (see heatsim2/curved_laminate_combined_surfcorr_2d.py for details)
    if (ExtraVolumeFactor > 1.0f) {
      ExtraVolumeFactor=1.0f;
    } 
    // ... Opposite limiting case: 
    if (ExtraVolumeFactor < -0.6f) {
      ExtraVolumeFactor=-0.6f;
    }
    
    Predicted_T = Predicted_T/(1.0f + ExtraVolumeFactor);

    // Consider heat flow along the line 
    if (avgcurvatures[avgcurvaturespos] >= 0.0f) {
      /* concave case */
      if ((iop_dx && (iop_dx[iop_dx_pos] != 0.0f || iop_dy[iop_dy_pos] != 0.0f)) && iop_dparallel > 0.2f*sqrt(4.0f*alpha_parallel*tvec[tpos]) && iop_dparallel > 0.2f*linelength[linelengthpos]) {
	// step size is large relative to length... use integrate_over_pixel method 
	Predicted_T *= (erf(((linelength[linelengthpos]+iop_dparallel/2.0f)/sqrt(4.0f*alpha_parallel*tvec[tpos]))*sqrt(1.0f + depth[depthpos]*avgcurvatures[avgcurvaturespos])*sqrt(deceleration(linelength[linelengthpos]*avgcurvatures[avgcurvaturespos],linelength[linelengthpos],depth[depthpos]))) - erf(((linelength[linelengthpos]-iop_dparallel/2.0f)/sqrt(4.0f*alpha_parallel*tvec[tpos]))*sqrt(1.0f + depth[depthpos]*avgcurvatures[avgcurvaturespos])*sqrt(deceleration(linelength[linelengthpos]*avgcurvatures[avgcurvaturespos],linelength[linelengthpos],depth[depthpos]))))/(2.0f*iop_dparallel);
      } else {
	Predicted_T *= exp(-(pow(linelength[linelengthpos],2.0f)/(4.0f*alpha_parallel*tvec[tpos]))*(1.0f + depth[depthpos]*avgcurvatures[avgcurvaturespos])*deceleration(linelength[linelengthpos]*avgcurvatures[avgcurvaturespos],linelength[linelengthpos],depth[depthpos]))/sqrt(4.0f*((float)M_PI)*alpha_parallel*tvec[tpos]);
      } 
    } else {
      /* convex case */
      if ((iop_dx && (iop_dx[iop_dx_pos] != 0.0f || iop_dy[iop_dy_pos] != 0.0f)) && iop_dparallel > 0.2f*sqrt(4.0f*alpha_parallel*tvec[tpos]) && iop_dparallel > 0.2f*linelength[linelengthpos]) {
	// step size is large relative to length... use integrate_over_pixel method.... don't use the inner_sin_sq term because 
	// it is hard to integrate... Use the perturbation approach instead. this is only for very short time, when the perturbation approach is pretty good...
	Predicted_T *= (erf(((linelength[linelengthpos]+iop_dparallel/2.0f)/sqrt(4.0f*alpha_parallel*tvec[tpos]))*sqrt(1.0f + depth[depthpos]*avgcurvatures[avgcurvaturespos])) - erf(((linelength[linelengthpos]-iop_dparallel/2.0f)/sqrt(4.0f*alpha_parallel*tvec[tpos]))*sqrt(1.0f + depth[depthpos]*avgcurvatures[avgcurvaturespos])))/(2.0f*iop_dparallel);
      } else {
	Predicted_T *= exp(-(pow((1.0f/fabs(avgcurvatures[avgcurvaturespos]))-depth[depthpos],2.0f)/(4.0f*alpha_parallel*tvec[tpos]))*(1.0f + depth[depthpos]/((1.0f/fabs(avgcurvatures[avgcurvaturespos]))-depth[depthpos]))*4.0f*inner_sin_sq(0.5f*linelength[linelengthpos]*avgcurvatures[avgcurvaturespos]))/sqrt(4.0f*((float)M_PI)*alpha_parallel*tvec[tpos]);
      }

    }
    
    // Consider heat flow across the line... here distance=0 so main factor would be exp(-0^2/(4alphat))
    if ((iop_dx && (iop_dx[iop_dx_pos] != 0.0f || iop_dy[iop_dy_pos] != 0.0f)) && iop_dparallel > 0.2f*sqrt(4.0f*alpha_parallel*tvec[tpos])) {
      /* integrate_over_pixel method */
      Predicted_T *= erf(iop_dcross/(2.0f*sqrt(4.0f*alpha_cross*tvec[tpos])))/iop_dcross;
    } else {
      Predicted_T /= sqrt(4.0f*((float)M_PI)*alpha_cross*tvec[tpos]);
    }
    sum+=Predicted_T;
    
  }
  
  result[itercnt]=sum;
  
}
