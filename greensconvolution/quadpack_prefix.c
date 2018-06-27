// regenerate qagse_fparams.c with:
// f2c -a qagse_fparams.f
// patch -p0 <qagse_fparams.patch
typedef unsigned long uint64_t;
typedef long int64_t;

#ifndef NULL
#define NULL ((void*)0l)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#ifdef static 
#undef static
#endif

#define static const __constant // f2c generates static when it means const

// Force to single precision (qagpe is basically single precision anyway)
typedef float doublereal;
typedef float real;
typedef int integer;
typedef int logical;


int assert(int a) {
  char *null=NULL;
  if (!a) { 
    if (*null) return 0;// attempt to read from invalid address zero
  }
  return 1;
 }

//typedef real (*E_fp)();
typedef char *E_fp;  // Don't use this anymore... hardwired to funct(...)

float dabs(float p) { return fabs(p); }
float dmax(float p,float q) { if (p > q) return p;else return q; }
float dmin(float p,float q) { if (p < q) return p;else return q; }
//float min(float p,float q) { if (p < q) return p;else return q; }


doublereal pow_dd(doublereal *arg1,const __constant doublereal *arg2)
{
  return pow(*arg1,*arg2);
}

/* C source for R1MACH -- remove the * in column 1 */
float r1mach_(const __constant integer *i)
{
	switch(*i){
	  case 1: return FLT_MIN;
	  case 2: return FLT_MAX;
	  case 3: return FLT_EPSILON/FLT_RADIX;
	  case 4: return FLT_EPSILON;
	  case 5: return log10((float)FLT_RADIX);
	  }
//  fprintf(stderr, "invalid argument: r1mach(%ld)\n", *i);
	assert(0); return 0; /* else complaint of missing return value */
}
#ifndef TRUE
#define TRUE (!0)
#endif

#ifndef FALSE
#define FALSE (0)
#endif
 


#define TRUE_ 1
#define FALSE_ 0

/* Insert qagse_fparams.c here, followed by quadpack.c */
