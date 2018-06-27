/* before this file we have:
 * quadpack_prefix.c
 * qagse_fparams.c 
 */

// regenerate qagse_fparams.c with:
// f2c -a qagse_fparams.f
// patch -p0 <qagse_fparams.patch

doublereal funct_(doublereal *x, doublereal *v, doublereal *a) 
{
  return pow(*x,-1.5f)*pow(1.0f-(*x),-1.5f)*exp(-(1.0f+(*a)*(*x))/((*v)*(*x)*(1.0f-(*x))));
}

#define LIMIT 50

__kernel void quadpack_opencl(
			      __global const float *zvec, __global const uint64_t *zvecshape,__global const uint64_t *zvecstrides,
			      __global const float *xvec, __global const uint64_t *xvecshape,__global const uint64_t *xvecstrides,
			      __global const float *tvec, __global const uint64_t *tvecshape,__global const uint64_t *tvecstrides,
			      float yval,
			      __global const uint64_t *sumstrides, __global const uint64_t *shape,
			      float alphaz,
			      float alphaxy,
			      __global float *result,__global const uint64_t *resultstrides,
			      float coeff,
			      __global const uint64_t *axissumflag,
			      uint64_t ndim)  
{
  uint64_t itercnt=get_global_id(0);

  int64_t zpos,tpos,xpos,resultpos,sumcnt,sumpos;
  uint64_t zposbase,tposbase,xposbase;
  float sum=0.0;
  int64_t axiscnt2,axispos;
  int loopdone=FALSE;
  int cnt;



  float epsilon;
  float my_a,my_v,my_c;
  float upper_bound;
  float epsabs,epsrel; // defaults from scipy
  float resultval,abserr;
  int neval;
  int ier;
  int limit;
  float alist[LIMIT];
  float blist[LIMIT];
  float rlist[LIMIT];
  float elist[LIMIT];
  int iord[LIMIT];
  int last;


  // itersum here was used for troubleshooting on AMD OpenCL
  // seems to have trouble due to irreproducibility in the call graph, 
  // but it doesn't error out... just give incorrect
  // results when used in looping mode (!)


  //float itersum=0.0;



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
      if (xvecshape[axiscnt2] > 1) {
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

    barrier(CLK_LOCAL_MEM_FENCE); /*** NOTE: Barriers are new and untested ***!!! */

    zpos=zposbase;
    tpos=tposbase;
    xpos=xposbase;
    sumpos=sumcnt;
    
    for (axiscnt2=0;axiscnt2 < ndim;axiscnt2++) {
      if (axissumflag[axiscnt2]) {
	axispos = sumpos/sumstrides[axiscnt2];
	sumpos -= axispos*sumstrides[axiscnt2];
      
	//itersum+=1.0;
	if (axispos >= *(((__global const int64_t *)shape)+axiscnt2)) {
	  //itersum+=100.0;
	  loopdone=TRUE;
	  break;
	}
	//sum+=9.9+sumcnt*45;
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
    if (sumpos > 0) {
      // itersum+=300.0;
      break;
    } 
    if (loopdone) break;

    //fprintf(stderr,"zpos=%d\n",(int)zpos);
    //fprintf(stderr,"xpos=%d\n",(int)xpos);
    //fprintf(stderr,"tpos=%d\n",(int)tpos);
  
    resultval=0.0;
    abserr=0.0;
    neval=0;
    ier=0;
    epsabs=1.e-6; // defaults from scipy
    epsrel=1.e-4;
    last=0;
    limit=LIMIT;

    my_c=fabs(sqrt(pow(xvec[xpos],2.0f)*(alphaz/alphaxy) + pow(yval,2.0f)*(alphaz/alphaxy) + pow(zvec[zpos],2.0f))/zvec[zpos]);
    //my_c=fabs(rconductivityscaledvec[xpos]/zvec[zpos]);
    my_a=my_c*my_c-1.0;
    my_v=(4*alphaz*tvec[tpos])/pow(zvec[zpos],2.0f);
  

    for (cnt=0; cnt < LIMIT;cnt++) {
      alist[cnt]=0.0;
      blist[cnt]=0.0;
      rlist[cnt]=0.0;
      elist[cnt]=0.0;
      iord[cnt]=0;
      
    }
  
    
    epsilon = 1.0f; 
    if (my_v > epsilon) {
      epsilon=my_v;
    } 
    if (my_a > epsilon) {
      epsilon=my_a;
    }
    epsilon = .05f/epsilon; 
    upper_bound=1.0f-epsilon;
    epsabs *= pow(my_v,2.0f);
    barrier(CLK_LOCAL_MEM_FENCE);

    qagse_(NULL,&my_v,&my_a,&epsilon,&upper_bound,&epsabs,&epsrel,&limit,&resultval,&abserr,&neval,&ier,alist,blist,rlist,elist,iord,&last);

    barrier(CLK_LOCAL_MEM_FENCE);
    
    resultval *= coeff/(pow(zvec[zpos],3.0f)*pow(my_v,2.0f));

    //sum+=0.1;


    //resultval=tvec[tpos];
    
    sum+=resultval; // +0.0*itersum;
    //itersum+=30+sumcnt*10;
  }
  result[itercnt]=sum;// *1.0+itersum*0.0; //itersum; // shape[1];
  
  
}
  
