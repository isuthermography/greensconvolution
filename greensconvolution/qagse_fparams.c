/* qagse_fparams.f -- translated by f2c (version 20100827).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

//#include "f2c.h"

/* Table of constant values */

static integer c__4 = 4;
static integer c__1 = 1;
static integer c__2 = 2;
static doublereal c_b42 = 1.5;

/*     This software in public domain because it was part of SLATEC */
/* Subroutine */ int qagse_(E_fp f, real *fp1, real *fp2, real *a, real *b, 
	real *epsabs, real *epsrel, integer *limit, real *result, real *
	abserr, integer *neval, integer *ier, real *alist__, real *blist, 
	real *rlist, real *elist, integer *iord, integer *last)
{
    /* System generated locals */
    integer i__1, i__2;
    real r__1, r__2;

    /* Local variables */
    integer k;
    real a1, a2, b1, b2;
    integer id;
    extern /* Subroutine */ int qk21_(E_fp, real *, real *, real *, real *, 
	    real *, real *, real *, real *);
    real area;
    extern /* Subroutine */ int qelg_(integer *, real *, real *, real *, real 
	    *, integer *);
    real dres;
    integer ksgn, nres;
    real area1, area2, area12, small, erro12;
    integer ierro;
    real defab1, defab2;
    integer ktmin, nrmax;
    real oflow, uflow;
    logical noext;
    extern /* Subroutine */ int qpsrt_(integer *, integer *, integer *, real *
	    , real *, integer *, integer *);
    extern doublereal r1mach_(const __constant integer *);
    integer iroff1, iroff2, iroff3;
    real res3la[3], error1, error2, rlist2[52];
    integer numrl2;
    real defabs, epmach, erlarg, abseps, correc, errbnd, resabs;
    integer jupbnd;
    real erlast, errmax;
    integer maxerr;
    real reseps;
    logical extrap;
    real ertest, errsum;

/* ***begin prologue  qagse */
/* ***date written   800101   (yymmdd) */
/* ***revision date  830518   (yymmdd) */
/* ***category no.  h2a1a1 */
/* ***keywords  automatic integrator, general-purpose, */
/*             (end point) singularities, extrapolation, */
/*             globally adaptive */
/* ***author  piessens,robert,appl. math. & progr. div. - k.u.leuven */
/*           de doncker,elise,appl. math. & progr. div. - k.u.leuven */
/* ***purpose  the routine calculates an approximation result to a given */
/*            definite integral i = integral of f over (a,b), */
/*            hopefully satisfying following claim for accuracy */
/*            abs(i-result).le.max(epsabs,epsrel*abs(i)). */
/* ***description */

/*        computation of a definite integral */
/*        standard fortran subroutine */
/*        real version */

/*        parameters */
/*         on entry */
/*            f      - real */
/*                     function subprogram defining the integrand */
/*                     function f(x). the actual name for f needs to be */
/*                     declared e x t e r n a l in the driver program. */
/*            fp1    - real  parameter #1 for f */
/*            fp2    - real  parameter #2 for f */

/*            a      - real */
/*                     lower limit of integration */

/*            b      - real */
/*                     upper limit of integration */

/*            epsabs - real */
/*                     absolute accuracy requested */
/*            epsrel - real */
/*                     relative accuracy requested */
/*                     if  epsabs.le.0 */
/*                     and epsrel.lt.max(50*rel.mach.acc.,0.5d-28), */
/*                     the routine will end with ier = 6. */

/*            limit  - integer */
/*                     gives an upperbound on the number of subintervals */
/*                     in the partition of (a,b) */

/*         on return */
/*            result - real */
/*                     approximation to the integral */

/*            abserr - real */
/*                     estimate of the modulus of the absolute error, */
/*                     which should equal or exceed abs(i-result) */

/*            neval  - integer */
/*                     number of integrand evaluations */

/*            ier    - integer */
/*                     ier = 0 normal and reliable termination of the */
/*                             routine. it is assumed that the requested */
/*                             accuracy has been achieved. */
/*                     ier.gt.0 abnormal termination of the routine */
/*                             the estimates for integral and error are */
/*                             less reliable. it is assumed that the */
/*                             requested accuracy has not been achieved. */
/*            error messages */
/*                         = 1 maximum number of subdivisions allowed */
/*                             has been achieved. one can allow more sub- */
/*                             divisions by increasing the value of limit */
/*                             (and taking the according dimension */
/*                             adjustments into account). however, if */
/*                             this yields no improvement it is advised */
/*                             to analyze the integrand in order to */
/*                             determine the integration difficulties. if */
/*                             the position of a local difficulty can be */
/*                             determined (e.g. singularity, */
/*                             discontinuity within the interval) one */
/*                             will probably gain from splitting up the */
/*                             interval at this point and calling the */
/*                             integrator on the subranges. if possible, */
/*                             an appropriate special-purpose integrator */
/*                             should be used, which is designed for */
/*                             handling the type of difficulty involved. */
/*                         = 2 the occurrence of roundoff error is detec- */
/*                             ted, which prevents the requested */
/*                             tolerance from being achieved. */
/*                             the error may be under-estimated. */
/*                         = 3 extremely bad integrand behaviour */
/*                             occurs at some points of the integration */
/*                             interval. */
/*                         = 4 the algorithm does not converge. */
/*                             roundoff error is detected in the */
/*                             extrapolation table. */
/*                             it is presumed that the requested */
/*                             tolerance cannot be achieved, and that the */
/*                             returned result is the best which can be */
/*                             obtained. */
/*                         = 5 the integral is probably divergent, or */
/*                             slowly convergent. it must be noted that */
/*                             divergence can occur with any other value */
/*                             of ier. */
/*                         = 6 the input is invalid, because */
/*                             epsabs.le.0 and */
/*                             epsrel.lt.max(50*rel.mach.acc.,0.5d-28). */
/*                             result, abserr, neval, last, rlist(1), */
/*                             iord(1) and elist(1) are set to zero. */
/*                             alist(1) and blist(1) are set to a and b */
/*                             respectively. */

/*            alist  - real */
/*                     vector of dimension at least limit, the first */
/*                      last  elements of which are the left end points */
/*                     of the subintervals in the partition of the */
/*                     given integration range (a,b) */

/*            blist  - real */
/*                     vector of dimension at least limit, the first */
/*                      last  elements of which are the right end points */
/*                     of the subintervals in the partition of the given */
/*                     integration range (a,b) */

/*            rlist  - real */
/*                     vector of dimension at least limit, the first */
/*                      last  elements of which are the integral */
/*                     approximations on the subintervals */

/*            elist  - real */
/*                     vector of dimension at least limit, the first */
/*                      last  elements of which are the moduli of the */
/*                     absolute error estimates on the subintervals */

/*            iord   - integer */
/*                     vector of dimension at least limit, the first k */
/*                     elements of which are pointers to the */
/*                     error estimates over the subintervals, */
/*                     such that elist(iord(1)), ..., elist(iord(k)) */
/*                     form a decreasing sequence, with k = last */
/*                     if last.le.(limit/2+2), and k = limit+1-last */
/*                     otherwise */

/*            last   - integer */
/*                     number of subintervals actually produced in the */
/*                     subdivision process */

/* ***references  (none) */
/* ***routines called  qelg,qk21,qpsrt,r1mach */
/* ***end prologue  qagse */




/*            the dimension of rlist2 is determined by the value of */
/*            limexp in subroutine qelg (rlist2 should be of dimension */
/*            (limexp+2) at least). */

/*            list of major variables */
/*            ----------------------- */

/*           alist     - list of left end points of all subintervals */
/*                       considered up to now */
/*           blist     - list of right end points of all subintervals */
/*                       considered up to now */
/*           rlist(i)  - approximation to the integral over */
/*                       (alist(i),blist(i)) */
/*           rlist2    - array of dimension at least limexp+2 */
/*                       containing the part of the epsilon table */
/*                       which is still needed for further computations */
/*           elist(i)  - error estimate applying to rlist(i) */
/*           maxerr    - pointer to the interval with largest error */
/*                       estimate */
/*           errmax    - elist(maxerr) */
/*           erlast    - error on the interval currently subdivided */
/*                       (before that subdivision has taken place) */
/*           area      - sum of the integrals over the subintervals */
/*           errsum    - sum of the errors over the subintervals */
/*           errbnd    - requested accuracy max(epsabs,epsrel* */
/*                       abs(result)) */
/*           *****1    - variable for the left interval */
/*           *****2    - variable for the right interval */
/*           last      - index for subdivision */
/*           nres      - number of calls to the extrapolation routine */
/*           numrl2    - number of elements currently in rlist2. if an */
/*                       appropriate approximation to the compounded */
/*                       integral has been obtained it is put in */
/*                       rlist2(numrl2) after numrl2 has been increased */
/*                       by one. */
/*           small     - length of the smallest interval considered */
/*                       up to now, multiplied by 1.5 */
/*           erlarg    - sum of the errors over the intervals larger */
/*                       than the smallest interval considered up to now */
/*           extrap    - logical variable denoting that the routine */
/*                       is attempting to perform extrapolation */
/*                       i.e. before subdividing the smallest interval */
/*                       we try to decrease the value of erlarg. */
/*           noext     - logical variable denoting that extrapolation */
/*                       is no longer allowed (true value) */

/*            machine dependent constants */
/*            --------------------------- */

/*           epmach is the largest relative spacing. */
/*           uflow is the smallest positive magnitude. */
/*           oflow is the largest positive magnitude. */

/* ***first executable statement  qagse */
    /* Parameter adjustments */
    --iord;
    --elist;
    --rlist;
    --blist;
    --alist__;

    /* Function Body */
    epmach = r1mach_(&c__4);

/*            test on validity of parameters */
/*            ------------------------------ */
    *ier = 0;
    *neval = 0;
    *last = 0;
    *result = 0.f;
    *abserr = 0.f;
    alist__[1] = *a;
    blist[1] = *b;
    rlist[1] = 0.f;
    elist[1] = 0.f;
/* Computing MAX */
    r__1 = epmach * 50.f;
    if (*epsabs <= 0.f && *epsrel < dmax(r__1,5e-15f)) {
	*ier = 6;
    }
    if (*ier == 6) {
	goto L999;
    }

/*           first approximation to the integral */
/*           ----------------------------------- */

    uflow = r1mach_(&c__1);
    oflow = r1mach_(&c__2);
    ierro = 0;
    qk21_((E_fp)f, fp1, fp2, a, b, result, abserr, &defabs, &resabs);

/*           test on accuracy. */

    dres = dabs(*result);
/* Computing MAX */
    r__1 = *epsabs, r__2 = *epsrel * dres;
    errbnd = dmax(r__1,r__2);
    *last = 1;
    rlist[1] = *result;
    elist[1] = *abserr;
    iord[1] = 1;
    if (*abserr <= epmach * 100.f * defabs && *abserr > errbnd) {
	*ier = 2;
    }
    if (*limit == 1) {
	*ier = 1;
    }
    if (*ier != 0 || (*abserr <= errbnd && *abserr != resabs) || *abserr == 0.f)
	     {
	goto L140;
    }

/*           initialization */
/*           -------------- */

    rlist2[0] = *result;
    errmax = *abserr;
    maxerr = 1;
    area = *result;
    errsum = *abserr;
    *abserr = oflow;
    nrmax = 1;
    nres = 0;
    numrl2 = 2;
    ktmin = 0;
    extrap = FALSE_;
    noext = FALSE_;
    iroff1 = 0;
    iroff2 = 0;
    iroff3 = 0;
    ksgn = -1;
    if (dres >= (1.f - epmach * 50.f) * defabs) {
	ksgn = 1;
    }

/*           main do-loop */
/*           ------------ */

    i__1 = *limit;
    for (*last = 2; *last <= i__1; ++(*last)) {

/*           bisect the subinterval with the nrmax-th largest */
/*           error estimate. */

	a1 = alist__[maxerr];
	b1 = (alist__[maxerr] + blist[maxerr]) * .5f;
	a2 = b1;
	b2 = blist[maxerr];
	erlast = errmax;
	qk21_((E_fp)f, fp1, fp2, &a1, &b1, &area1, &error1, &resabs, &defab1);
	qk21_((E_fp)f, fp1, fp2, &a2, &b2, &area2, &error2, &resabs, &defab2);

/*           improve previous approximations to integral */
/*           and error and test for accuracy. */

	area12 = area1 + area2;
	erro12 = error1 + error2;
	errsum = errsum + erro12 - errmax;
	area = area + area12 - rlist[maxerr];
	if (defab1 == error1 || defab2 == error2) {
	    goto L15;
	}
	if ((r__1 = rlist[maxerr] - area12, dabs(r__1)) > dabs(area12) * 
		1e-5f || erro12 < errmax * .99f) {
	    goto L10;
	}
	if (extrap) {
	    ++iroff2;
	}
	if (! extrap) {
	    ++iroff1;
	}
L10:
	if (*last > 10 && erro12 > errmax) {
	    ++iroff3;
	}
L15:
	rlist[maxerr] = area1;
	rlist[*last] = area2;
/* Computing MAX */
	r__1 = *epsabs, r__2 = *epsrel * dabs(area);
	errbnd = dmax(r__1,r__2);

/*           test for roundoff error and eventually */
/*           set error flag. */

	if (iroff1 + iroff2 >= 10 || iroff3 >= 20) {
	    *ier = 2;
	}
	if (iroff2 >= 5) {
	    ierro = 3;
	}

/*           set error flag in the case that the number of */
/*           subintervals equals limit. */

	if (*last == *limit) {
	    *ier = 1;
	}

/*           set error flag in the case of bad integrand behaviour */
/*           at a point of the integration range. */

/* Computing MAX */
	r__1 = dabs(a1), r__2 = dabs(b2);
	if (dmax(r__1,r__2) <= (epmach * 100.f + 1.f) * (dabs(a2) + uflow * 
		1e3f)) {
	    *ier = 4;
	}

/*           append the newly-created intervals to the list. */

	if (error2 > error1) {
	    goto L20;
	}
	alist__[*last] = a2;
	blist[maxerr] = b1;
	blist[*last] = b2;
	elist[maxerr] = error1;
	elist[*last] = error2;
	goto L30;
L20:
	alist__[maxerr] = a2;
	alist__[*last] = a1;
	blist[*last] = b1;
	rlist[maxerr] = area2;
	rlist[*last] = area1;
	elist[maxerr] = error2;
	elist[*last] = error1;

/*           call subroutine qpsrt to maintain the descending ordering */
/*           in the list of error estimates and select the */
/*           subinterval with nrmax-th largest error estimate (to be */
/*           bisected next). */

L30:
	qpsrt_(limit, last, &maxerr, &errmax, &elist[1], &iord[1], &nrmax);
/* ***jump out of do-loop */
	if (errsum <= errbnd) {
	    goto L115;
	}
/* ***jump out of do-loop */
	if (*ier != 0) {
	    goto L100;
	}
	if (*last == 2) {
	    goto L80;
	}
	if (noext) {
	    goto L90;
	}
	erlarg -= erlast;
	if ((r__1 = b1 - a1, dabs(r__1)) > small) {
	    erlarg += erro12;
	}
	if (extrap) {
	    goto L40;
	}

/*           test whether the interval to be bisected next is the */
/*           smallest interval. */

	if ((r__1 = blist[maxerr] - alist__[maxerr], dabs(r__1)) > small) {
	    goto L90;
	}
	extrap = TRUE_;
	nrmax = 2;
L40:
	if (ierro == 3 || erlarg <= ertest) {
	    goto L60;
	}

/*           the smallest interval has the largest error. */
/*           before bisecting decrease the sum of the errors */
/*           over the larger intervals (erlarg) and perform */
/*           extrapolation. */

	id = nrmax;
	jupbnd = *last;
	if (*last > *limit / 2 + 2) {
	    jupbnd = *limit + 3 - *last;
	}
	i__2 = jupbnd;
	for (k = id; k <= i__2; ++k) {
	    maxerr = iord[nrmax];
	    errmax = elist[maxerr];
/* ***jump out of do-loop */
	    if ((r__1 = blist[maxerr] - alist__[maxerr], dabs(r__1)) > small) 
		    {
		goto L90;
	    }
	    ++nrmax;
/* L50: */
	}

/*           perform extrapolation. */

L60:
	++numrl2;
	rlist2[numrl2 - 1] = area;
	qelg_(&numrl2, rlist2, &reseps, &abseps, res3la, &nres);
	++ktmin;
	if (ktmin > 5 && *abserr < errsum * .001f) {
	    *ier = 5;
	}
	if (abseps >= *abserr) {
	    goto L70;
	}
	ktmin = 0;
	*abserr = abseps;
	*result = reseps;
	correc = erlarg;
/* Computing MAX */
	r__1 = *epsabs, r__2 = *epsrel * dabs(reseps);
	ertest = dmax(r__1,r__2);
/* ***jump out of do-loop */
	if (*abserr <= ertest) {
	    goto L100;
	}

/*           prepare bisection of the smallest interval. */

L70:
	if (numrl2 == 1) {
	    noext = TRUE_;
	}
	if (*ier == 5) {
	    goto L100;
	}
	maxerr = iord[1];
	errmax = elist[maxerr];
	nrmax = 1;
	extrap = FALSE_;
	small *= .5f;
	erlarg = errsum;
	goto L90;
L80:
	small = (r__1 = *b - *a, dabs(r__1)) * .375f;
	erlarg = errsum;
	ertest = errbnd;
	rlist2[1] = area;
L90:
	;
    }

/*           set final result and error estimate. */
/*           ------------------------------------ */

L100:
    if (*abserr == oflow) {
	goto L115;
    }
    if (*ier + ierro == 0) {
	goto L110;
    }
    if (ierro == 3) {
	*abserr += correc;
    }
    if (*ier == 0) {
	*ier = 3;
    }
    if (*result != 0.f && area != 0.f) {
	goto L105;
    }
    if (*abserr > errsum) {
	goto L115;
    }
    if (area == 0.f) {
	goto L130;
    }
    goto L110;
L105:
    if (*abserr / dabs(*result) > errsum / dabs(area)) {
	goto L115;
    }

/*           test on divergence. */

L110:
/* Computing MAX */
    r__1 = dabs(*result), r__2 = dabs(area);
    if (ksgn == -1 && dmax(r__1,r__2) <= defabs * .01f) {
	goto L130;
    }
    if (.01f > *result / area || *result / area > 100.f || errsum > dabs(area)
	    ) {
	*ier = 6;
    }
    goto L130;

/*           compute global integral sum. */

L115:
    *result = 0.f;
    i__1 = *last;
    for (k = 1; k <= i__1; ++k) {
	*result += rlist[k];
/* L120: */
    }
    *abserr = errsum;
L130:
    if (*ier > 2) {
	--(*ier);
    }
L140:
    *neval = *last * 42 - 21;
L999:
    return 0;
} /* qagse_ */

/* Subroutine */ int qelg_(integer *n, real *epstab, real *result, real *
	abserr, real *res3la, integer *nres)
{
    /* System generated locals */
    integer i__1;
    real r__1, r__2, r__3;

    /* Local variables */
    integer i__;
    real e0, e1, e2, e3;
    integer k1, k2, k3, ib, ie;
    real ss;
    integer ib2;
    real res;
    integer num;
    real err1, err2, err3, tol1, tol2, tol3;
    integer indx;
    real e1abs, oflow, error, delta1, delta2, delta3;
    extern doublereal r1mach_(const __constant integer *);
    real epmach, epsinf;
    integer newelm, limexp;

/* ***begin prologue  qelg */
/* ***refer to  qagie,qagoe,qagpe,qagse */
/* ***routines called  r1mach */
/* ***revision date  830518   (yymmdd) */
/* ***keywords  epsilon algorithm, convergence acceleration, */
/*             extrapolation */
/* ***author  piessens,robert,appl. math. & progr. div. - k.u.leuven */
/*           de doncker,elise,appl. math & progr. div. - k.u.leuven */
/* ***purpose  the routine determines the limit of a given sequence of */
/*            approximations, by means of the epsilon algorithm of */
/*            p. wynn. an estimate of the absolute error is also given. */
/*            the condensed epsilon table is computed. only those */
/*            elements needed for the computation of the next diagonal */
/*            are preserved. */
/* ***description */

/*           epsilon algorithm */
/*           standard fortran subroutine */
/*           real version */

/*           parameters */
/*              n      - integer */
/*                       epstab(n) contains the new element in the */
/*                       first column of the epsilon table. */

/*              epstab - real */
/*                       vector of dimension 52 containing the elements */
/*                       of the two lower diagonals of the triangular */
/*                       epsilon table. the elements are numbered */
/*                       starting at the right-hand corner of the */
/*                       triangle. */

/*              result - real */
/*                       resulting approximation to the integral */

/*              abserr - real */
/*                       estimate of the absolute error computed from */
/*                       result and the 3 previous results */

/*              res3la - real */
/*                       vector of dimension 3 containing the last 3 */
/*                       results */

/*              nres   - integer */
/*                       number of calls to the routine */
/*                       (should be zero at first call) */

/* ***end prologue  qelg */


/*           list of major variables */
/*           ----------------------- */

/*           e0     - the 4 elements on which the */
/*           e1       computation of a new element in */
/*           e2       the epsilon table is based */
/*           e3                 e0 */
/*                        e3    e1    new */
/*                              e2 */
/*           newelm - number of elements to be computed in the new */
/*                    diagonal */
/*           error  - error = abs(e1-e0)+abs(e2-e1)+abs(new-e2) */
/*           result - the element in the new diagonal with least value */
/*                    of error */

/*           machine dependent constants */
/*           --------------------------- */

/*           epmach is the largest relative spacing. */
/*           oflow is the largest positive magnitude. */
/*           limexp is the maximum number of elements the epsilon */
/*           table can contain. if this number is reached, the upper */
/*           diagonal of the epsilon table is deleted. */

/* ***first executable statement  qelg */
    /* Parameter adjustments */
    --res3la;
    --epstab;

    /* Function Body */
    epmach = r1mach_(&c__4);
    oflow = r1mach_(&c__2);
    ++(*nres);
    *abserr = oflow;
    *result = epstab[*n];
    if (*n < 3) {
	goto L100;
    }
    limexp = 50;
    epstab[*n + 2] = epstab[*n];
    newelm = (*n - 1) / 2;
    epstab[*n] = oflow;
    num = *n;
    k1 = *n;
    i__1 = newelm;
    for (i__ = 1; i__ <= i__1; ++i__) {
	k2 = k1 - 1;
	k3 = k1 - 2;
	res = epstab[k1 + 2];
	e0 = epstab[k3];
	e1 = epstab[k2];
	e2 = res;
	e1abs = dabs(e1);
	delta2 = e2 - e1;
	err2 = dabs(delta2);
/* Computing MAX */
	r__1 = dabs(e2);
	tol2 = dmax(r__1,e1abs) * epmach;
	delta3 = e1 - e0;
	err3 = dabs(delta3);
/* Computing MAX */
	r__1 = e1abs, r__2 = dabs(e0);
	tol3 = dmax(r__1,r__2) * epmach;
	if (err2 > tol2 || err3 > tol3) {
	    goto L10;
	}

/*           if e0, e1 and e2 are equal to within machine */
/*           accuracy, convergence is assumed. */
/*           result = e2 */
/*           abserr = abs(e1-e0)+abs(e2-e1) */

	*result = res;
	*abserr = err2 + err3;
/* ***jump out of do-loop */
	goto L100;
L10:
	e3 = epstab[k1];
	epstab[k1] = e1;
	delta1 = e1 - e3;
	err1 = dabs(delta1);
/* Computing MAX */
	r__1 = e1abs, r__2 = dabs(e3);
	tol1 = dmax(r__1,r__2) * epmach;

/*           if two elements are very close to each other, omit */
/*           a part of the table by adjusting the value of n */

	if (err1 <= tol1 || err2 <= tol2 || err3 <= tol3) {
	    goto L20;
	}
	ss = 1.f / delta1 + 1.f / delta2 - 1.f / delta3;
	epsinf = (r__1 = ss * e1, dabs(r__1));

/*           test to detect irregular behaviour in the table, and */
/*           eventually omit a part of the table adjusting the value */
/*           of n. */

	if (epsinf > 1e-4f) {
	    goto L30;
	}
L20:
	*n = i__ + i__ - 1;
/* ***jump out of do-loop */
	goto L50;

/*           compute a new element and eventually adjust */
/*           the value of result. */

L30:
	res = e1 + 1.f / ss;
	epstab[k1] = res;
	k1 += -2;
	error = err2 + (r__1 = res - e2, dabs(r__1)) + err3;
	if (error > *abserr) {
	    goto L40;
	}
	*abserr = error;
	*result = res;
L40:
	;
    }

/*           shift the table. */

L50:
    if (*n == limexp) {
	*n = (limexp / 2 << 1) - 1;
    }
    ib = 1;
    if (num / 2 << 1 == num) {
	ib = 2;
    }
    ie = newelm + 1;
    i__1 = ie;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ib2 = ib + 2;
	epstab[ib] = epstab[ib2];
	ib = ib2;
/* L60: */
    }
    if (num == *n) {
	goto L80;
    }
    indx = num - *n + 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	epstab[i__] = epstab[indx];
	++indx;
/* L70: */
    }
L80:
    if (*nres >= 4) {
	goto L90;
    }
    res3la[*nres] = *result;
    *abserr = oflow;
    goto L100;

/*           compute error estimate */

L90:
    *abserr = (r__1 = *result - res3la[3], dabs(r__1)) + (r__2 = *result - 
	    res3la[2], dabs(r__2)) + (r__3 = *result - res3la[1], dabs(r__3));
    res3la[1] = res3la[2];
    res3la[2] = res3la[3];
    res3la[3] = *result;
L100:
/* Computing MAX */
    r__1 = *abserr, r__2 = epmach * 5.f * dabs(*result);
    *abserr = dmax(r__1,r__2);
    return 0;
} /* qelg_ */


    const __constant real xgk[11] = { .9956571630258081f,.9739065285171717f,
	    .9301574913557082f,.8650633666889845f,.7808177265864169f,
	    .6794095682990244f,.5627571346686047f,.4333953941292472f,
	    .2943928627014602f,.1488743389816312f,0.f };
    const __constant real wgk[11] = { .01169463886737187f,.03255816230796473f,
	    .054755896574352f,.07503967481091995f,.09312545458369761f,
	    .1093871588022976f,.1234919762620659f,.1347092173114733f,
	    .1427759385770601f,.1477391049013385f,.1494455540029169f };
    const __constant real wg[5] = { .06667134430868814f,.1494513491505806f,
	    .219086362515982f,.2692667193099964f,.2955242247147529f };


/* Subroutine */ int qk21_(E_fp f, real *fp1, real *fp2, real *a, real *b, 
	real *result, real *abserr, real *resabs, real *resasc)
{
    /* Initialized data */

    /* System generated locals */
    real r__1, r__2;
    doublereal d__1;

    /* Builtin functions */
    doublereal pow_dd(doublereal *, const __constant doublereal *);

    /* Local variables */
    integer j;
    real fc, fv1[10], fv2[10];
    integer jtw;
    real absc, resg, resk, fsum, fval1, fval2;
    integer jtwm1;
    real hlgth, centr, reskh;
    extern doublereal funct_(real *, real *, real *);
    real uflow;
    extern doublereal r1mach_(const __constant integer *);
    real epmach, dhlgth;

/* ***begin prologue  qk21 */
/* ***date written   800101   (yymmdd) */
/* ***revision date  830518   (yymmdd) */
/* ***category no.  h2a1a2 */
/* ***keywords  21-point gauss-kronrod rules */
/* ***author  piessens,robert,appl. math. & progr. div. - k.u.leuven */
/*           de doncker,elise,appl. math. & progr. div. - k.u.leuven */
/* ***purpose  to compute i = integral of f over (a,b), with error */
/*                           estimate */
/*                       j = integral of abs(f) over (a,b) */
/* ***description */

/*           integration rules */
/*           standard fortran subroutine */
/*           real version */

/*           parameters */
/*            on entry */
/*              f      - real */
/*                       function subprogram defining the integrand */
/*                       function f(x). the actual name for f needs to be */
/*                       declared e x t e r n a l in the driver program. */

/*              a      - real */
/*                       lower limit of integration */

/*              b      - real */
/*                       upper limit of integration */

/*            on return */
/*              result - real */
/*                       approximation to the integral i */
/*                       result is computed by applying the 21-point */
/*                       kronrod rule (resk) obtained by optimal addition */
/*                       of abscissae to the 10-point gauss rule (resg). */

/*              abserr - real */
/*                       estimate of the modulus of the absolute error, */
/*                       which should not exceed abs(i-result) */

/*              resabs - real */
/*                       approximation to the integral j */

/*              resasc - real */
/*                       approximation to the integral of abs(f-i/(b-a)) */
/*                       over (a,b) */

/* ***references  (none) */
/* ***routines called  r1mach */
/* ***end prologue  qk21 */



/*           the abscissae and weights are given for the interval (-1,1). */
/*           because of symmetry only the positive abscissae and their */
/*           corresponding weights are given. */

/*           xgk    - abscissae of the 21-point kronrod rule */
/*                    xgk(2), xgk(4), ...  abscissae of the 10-point */
/*                    gauss rule */
/*                    xgk(1), xgk(3), ...  abscissae which are optimally */
/*                    added to the 10-point gauss rule */

/*           wgk    - weights of the 21-point kronrod rule */

/*           wg     - weights of the 10-point gauss rule */





/*           list of major variables */
/*           ----------------------- */

/*           centr  - mid point of the interval */
/*           hlgth  - half-length of the interval */
/*           absc   - abscissa */
/*           fval*  - function value */
/*           resg   - result of the 10-point gauss formula */
/*           resk   - result of the 21-point kronrod formula */
/*           reskh  - approximation to the mean value of f over (a,b), */
/*                    i.e. to i/(b-a) */


/*           machine dependent constants */
/*           --------------------------- */

/*           epmach is the largest relative spacing. */
/*           uflow is the smallest positive magnitude. */

/* ***first executable statement  qk21 */
    epmach = r1mach_(&c__4);
    uflow = r1mach_(&c__1);

    centr = (*a + *b) * .5f;
    hlgth = (*b - *a) * .5f;
    dhlgth = dabs(hlgth);

/*           compute the 21-point kronrod approximation to */
/*           the integral, and estimate the absolute error. */

    resg = 0.f;
    fc = funct_(&centr, fp1, fp2);
    resk = wgk[10] * fc;
    *resabs = dabs(resk);
    for (j = 1; j <= 5; ++j) {
	jtw = j << 1;
	absc = hlgth * xgk[jtw - 1];
	r__1 = centr - absc;
	fval1 = funct_(&r__1, fp1, fp2);
	r__1 = centr + absc;
	fval2 = funct_(&r__1, fp1, fp2);
	fv1[jtw - 1] = fval1;
	fv2[jtw - 1] = fval2;
	fsum = fval1 + fval2;
	resg += wg[j - 1] * fsum;
	resk += wgk[jtw - 1] * fsum;
	*resabs += wgk[jtw - 1] * (dabs(fval1) + dabs(fval2));
/* L10: */
    }
    for (j = 1; j <= 5; ++j) {
	jtwm1 = (j << 1) - 1;
	absc = hlgth * xgk[jtwm1 - 1];
	r__1 = centr - absc;
	fval1 = funct_(&r__1, fp1, fp2);
	r__1 = centr + absc;
	fval2 = funct_(&r__1, fp1, fp2);
	fv1[jtwm1 - 1] = fval1;
	fv2[jtwm1 - 1] = fval2;
	fsum = fval1 + fval2;
	resk += wgk[jtwm1 - 1] * fsum;
	*resabs += wgk[jtwm1 - 1] * (dabs(fval1) + dabs(fval2));
/* L15: */
    }
    reskh = resk * .5f;
    *resasc = wgk[10] * (r__1 = fc - reskh, dabs(r__1));
    for (j = 1; j <= 10; ++j) {
	*resasc += wgk[j - 1] * ((r__1 = fv1[j - 1] - reskh, dabs(r__1)) + (
		r__2 = fv2[j - 1] - reskh, dabs(r__2)));
/* L20: */
    }
    *result = resk * hlgth;
    *resabs *= dhlgth;
    *resasc *= dhlgth;
    *abserr = (r__1 = (resk - resg) * hlgth, dabs(r__1));
    if (*resasc != 0.f && *abserr != 0.f) {
/* Computing MIN */
	d__1 = (doublereal) (*abserr * 200.f / *resasc);
	r__1 = 1.f, r__2 = pow_dd(&d__1, &c_b42);
	*abserr = *resasc * dmin(r__1,r__2);
    }
    if (*resabs > uflow / (epmach * 50.f)) {
/* Computing MAX */
	r__1 = epmach * 50.f * *resabs;
	*abserr = dmax(r__1,*abserr);
    }
    return 0;
} /* qk21_ */

/* Subroutine */ int qpsrt_(integer *limit, integer *last, integer *maxerr, 
	real *ermax, real *elist, integer *iord, integer *nrmax)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__, j, k, ido, ibeg, jbnd, isucc, jupbn;
    real errmin, errmax;

/* ***begin prologue  qpsrt */
/* ***refer to  qage,qagie,qagpe,qagse,qawce,qawse,qawoe */
/* ***routines called  (none) */
/* ***keywords  sequential sorting */
/* ***description */

/* 1.        qpsrt */
/*           ordering routine */
/*              standard fortran subroutine */
/*              real version */

/* 2.        purpose */
/*              this routine maintains the descending ordering */
/*              in the list of the local error estimates resulting from */
/*              the interval subdivision process. at each call two error */
/*              estimates are inserted using the sequential search */
/*              method, top-down for the largest error estimate */
/*              and bottom-up for the smallest error estimate. */

/* 3.        calling sequence */
/*              call qpsrt(limit,last,maxerr,ermax,elist,iord,nrmax) */

/*           parameters (meaning at output) */
/*              limit  - integer */
/*                       maximum number of error estimates the list */
/*                       can contain */

/*              last   - integer */
/*                       number of error estimates currently */
/*                       in the list */

/*              maxerr - integer */
/*                       maxerr points to the nrmax-th largest error */
/*                       estimate currently in the list */

/*              ermax  - real */
/*                       nrmax-th largest error estimate */
/*                       ermax = elist(maxerr) */

/*              elist  - real */
/*                       vector of dimension last containing */
/*                       the error estimates */

/*              iord   - integer */
/*                       vector of dimension last, the first k */
/*                       elements of which contain pointers */
/*                       to the error estimates, such that */
/*                       elist(iord(1)),... , elist(iord(k)) */
/*                       form a decreasing sequence, with */
/*                       k = last if last.le.(limit/2+2), and */
/*                       k = limit+1-last otherwise */

/*              nrmax  - integer */
/*                       maxerr = iord(nrmax) */

/* 4.        no subroutines or functions needed */
/* ***end prologue  qpsrt */


/*           check whether the list contains more than */
/*           two error estimates. */

/* ***first executable statement  qpsrt */
    /* Parameter adjustments */
    --iord;
    --elist;

    /* Function Body */
    if (*last > 2) {
	goto L10;
    }
    iord[1] = 1;
    iord[2] = 2;
    goto L90;

/*           this part of the routine is only executed */
/*           if, due to a difficult integrand, subdivision */
/*           increased the error estimate. in the normal case */
/*           the insert procedure should start after the */
/*           nrmax-th largest error estimate. */

L10:
    errmax = elist[*maxerr];
    if (*nrmax == 1) {
	goto L30;
    }
    ido = *nrmax - 1;
    i__1 = ido;
    for (i__ = 1; i__ <= i__1; ++i__) {
	isucc = iord[*nrmax - 1];
/* ***jump out of do-loop */
	if (errmax <= elist[isucc]) {
	    goto L30;
	}
	iord[*nrmax] = isucc;
	--(*nrmax);
/* L20: */
    }

/*           compute the number of elements in the list to */
/*           be maintained in descending order. this number */
/*           depends on the number of subdivisions still */
/*           allowed. */

L30:
    jupbn = *last;
    if (*last > *limit / 2 + 2) {
	jupbn = *limit + 3 - *last;
    }
    errmin = elist[*last];

/*           insert errmax by traversing the list top-down, */
/*           starting comparison from the element elist(iord(nrmax+1)). */

    jbnd = jupbn - 1;
    ibeg = *nrmax + 1;
    if (ibeg > jbnd) {
	goto L50;
    }
    i__1 = jbnd;
    for (i__ = ibeg; i__ <= i__1; ++i__) {
	isucc = iord[i__];
/* ***jump out of do-loop */
	if (errmax >= elist[isucc]) {
	    goto L60;
	}
	iord[i__ - 1] = isucc;
/* L40: */
    }
L50:
    iord[jbnd] = *maxerr;
    iord[jupbn] = *last;
    goto L90;

/*           insert errmin by traversing the list bottom-up. */

L60:
    iord[i__ - 1] = *maxerr;
    k = jbnd;
    i__1 = jbnd;
    for (j = i__; j <= i__1; ++j) {
	isucc = iord[k];
/* ***jump out of do-loop */
	if (errmin < elist[isucc]) {
	    goto L80;
	}
	iord[k + 1] = isucc;
	--k;
/* L70: */
    }
    iord[i__] = *last;
    goto L90;
L80:
    iord[k + 1] = *last;

/*           set maxerr and ermax. */

L90:
    *maxerr = iord[*nrmax];
    *ermax = elist[*maxerr];
    return 0;
} /* qpsrt_ */

