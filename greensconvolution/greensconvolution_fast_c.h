void greensconvolution_integrate_anisotropic_c(
					       const float *vrange,uint64_t nvrange, //number of rows in integraleval
					       const float *crange,uint64_t ncrange, // number of cols in integraleval
					       const float *integraleval,
					       const float *integral_dintegranddveval, // same size as integraleval
					       const float *integral_dintegranddceval, // same size as integraleval
					       const float *zvec, const uint64_t *zvecshape,const uint64_t *zvecstrides,
					       const float *xvec, const uint64_t *xvecshape,const uint64_t *xvecstrides,
					       const float *tvec, const uint64_t *tvecshape, const uint64_t *tvecstrides,
					       float yval,
					       const float *curvaturevec, const uint64_t *curvaturevecshape, const uint64_t *curvaturevecstrides,
					       const uint64_t *sumstrides,const uint64_t *shape,
					       float alphaz, float alphaxy, int8_t curvature_flag,
					       float *result,const uint64_t *resultstrides,
					       float coeff,
					       const uint64_t *axissumflag,
					       uint64_t ndim);
