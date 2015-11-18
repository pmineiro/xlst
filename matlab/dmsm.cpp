#include "mex.h"
#include "matrix.h"
#include <cmath>
#include <cstdint>
#include <thread>

/* % dense matrix times sparse matrix
 * >> A=sprandn(100,99,0.1); W=ones(1,99); B=randn(5,100); Z=B*A*diag(W); ZZ=dmsm(B,A,W); disp(norm(Z-ZZ))
 *      0
 * % dense matrix times scaled sparse matrix
 * >> A=sprandn(100,99,0.1); W=randn(1,99); B=randn(5,100); Z=B*A*diag(W); ZZ=dmsm(B,A,W); disp(norm(Z-ZZ))
      9.1859e-15
 * % dense matrix times scaled sparse matrix over column range
 * >> A=sprandn(100,99,0.1); W=randn(1,99); B=randn(5,100); Z=B*A(:,3:51)*diag(W(3:51)); ZZ=dmsm(B,A,W,3,51); disp(norm(Z-ZZ))
      5.3934e-15
 * % ibid, but with mask
 * >> A=sprandn(100,99,0.1); W=randn(1,99); B=randn(5,100); mask=3:51; Z=B*A(:,mask)*diag(W(mask)); ZZ=dmsm(B,A,W,mask); disp(norm(Z-ZZ))
      6.7228e-15
 */

/*
 * compilation, at matlab prompt: (adjust NUM_THREADS as appropriate)
 * 
 * == windows ==
 * 
 * mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=4 dmsm.cpp
 * 
 * == linux ==
 * 
 * mex OPTIMFLAGS="-O3" -largeArrayDims -DNUM_THREADS=4 -v CXXFLAGS='$CXXFLAGS -std=c++0x -fPIC' dmsm.cpp
 */


#define DENSE_MATRIX_PARAMETER_IN     prhs[0]
#define SPARSE_MATRIX_PARAMETER_IN    prhs[1]
#define DIAG_VECTOR_PARAMETER_IN      prhs[2]
#define MASK_PARAMETER_IN            prhs[3]
#define END_PARAMETER_IN              prhs[4]

// X = D*S*diag(W) => X' = diag(W)*S'*D'

template<typename scalar, 
         typename diagvector,
         typename range>
static void
sparsetic_times_densetic_impl (int nrhs, const mxArray* prhs[], mxArray* plhs[], 
                               size_t start, size_t end, const range& r, const diagvector& w)
{
  mwIndex* ir = mxGetIr(SPARSE_MATRIX_PARAMETER_IN);  /* Row indexing      */
  mwIndex* jc = mxGetJc(SPARSE_MATRIX_PARAMETER_IN);  /* Column count      */
  double* s  = mxGetPr(SPARSE_MATRIX_PARAMETER_IN);   /* Non-zero elements */
  scalar* Btic = (scalar*) mxGetData(DENSE_MATRIX_PARAMETER_IN);
  mwSize Bcol = mxGetM(DENSE_MATRIX_PARAMETER_IN);
  scalar* Xtic = (scalar*) mxGetData(plhs[0]);
  mwSize Xcol = mxGetM(plhs[0]);

  for (size_t ii=start; ii<end; ++ii) {          /* Loop through rows of A (and X) */
    size_t i (r[ii]);
    auto wi (w[i]);
    mwIndex stop = jc[i+1];
    for (mwIndex k=jc[i]; k<stop; ++k) {    /* Loop through non-zeros in ith row of A */
      double sk = wi * s[k];
      scalar* Bticrow = Btic + ir[k] * Bcol;
      scalar* Xticrow = Xtic + ii * Xcol;
      for (mwSize j=0; j<Xcol; ++j) {
        Xticrow[j] += sk * Bticrow[j];
      }
    }
  }
}

struct OneVector {
  inline double operator[] (int) const { return 1.0; } 
};

class FloatVector {
  private:
    float* p;
  public:
    FloatVector(float* _p) : p (_p) { }

    inline float operator[] (int i) const { return p[i]; }
};

class DoubleVector {
  private:
    double* p;
  public:
    DoubleVector(double* _p) : p (_p) { }

    inline double operator[] (int i) const { return p[i]; }
};

class SequenceRange {
  private:
    size_t start;
    size_t end;
  public:
    SequenceRange(size_t _start, size_t _end) : start(_start), end(_end) { }
        
    inline size_t operator[] (size_t i) const { return start+i; }
};

class ExplicitRange {
  private:
    double* p;
    size_t n;
  public:
    ExplicitRange(double* _p, size_t _n) : p(_p), n(_n) { }
        
    inline size_t operator[] (size_t i) const { return p[i]-1; }
};

template<typename scalar,
         typename range>
static void
sparsetic_times_densetic_impl3(int nrhs, const mxArray* prhs[], mxArray* plhs[], 
                               size_t start, size_t end, const range& r)
{       
  if (nrhs == 2 || mxIsEmpty(DIAG_VECTOR_PARAMETER_IN))
    {
      return sparsetic_times_densetic_impl<scalar> (nrhs, prhs, plhs, start, end, r, OneVector ());
    }
  else if (mxIsSingle(DIAG_VECTOR_PARAMETER_IN))
    { 
      return sparsetic_times_densetic_impl<scalar> (nrhs, prhs, plhs, start, end, r, FloatVector((float*)mxGetData(DIAG_VECTOR_PARAMETER_IN)));
    }
  else
    {
      return sparsetic_times_densetic_impl<scalar> (nrhs, prhs, plhs, start, end, r,DoubleVector((double*)mxGetData(DIAG_VECTOR_PARAMETER_IN)));
    }
}

template<typename scalar>
static void
sparsetic_times_densetic_impl2(int nrhs, const mxArray* prhs[], mxArray* plhs[], 
                               size_t start, size_t end)
{
  switch (nrhs) {
    case 4: {
      ExplicitRange r((double*) mxGetData(MASK_PARAMETER_IN),
                       mxGetNumberOfElements(MASK_PARAMETER_IN));
      return sparsetic_times_densetic_impl3<scalar> (nrhs, prhs, plhs, start, end, r);
    }
    
      break;
    default: {
      size_t Arow = mxGetN(SPARSE_MATRIX_PARAMETER_IN);
      size_t mystart = (nrhs >= 4) ? mxGetScalar(MASK_PARAMETER_IN) : 1;
      size_t myend = (nrhs == 5) ? mxGetScalar(END_PARAMETER_IN) : Arow;  
      SequenceRange r(mystart-1, myend);
  
      return sparsetic_times_densetic_impl3<scalar> (nrhs, prhs, plhs, start, end, r);
    }
      break;
  }
}

static void
sparsetic_times_densetic (int nrhs, const mxArray* prhs[], mxArray* plhs[], size_t start, size_t end)
{  
  return mxIsSingle(DENSE_MATRIX_PARAMETER_IN) 
    ? sparsetic_times_densetic_impl2<float> (nrhs, prhs, plhs, start, end)
    : sparsetic_times_densetic_impl2<double> (nrhs, prhs, plhs, start, end);
}

static int first = 1;

void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray*prhs[] ) { 
  if (first) {
    mexPrintf("dmsm using NUM_THREADS=%u\n",NUM_THREADS);
    mexEvalString("drawnow;");
    first=0;
  }

  switch (nrhs) {
    case 5:
      if (mxGetM(END_PARAMETER_IN) != 1 || mxGetN(END_PARAMETER_IN) != 1) {
        mexErrMsgTxt("End must be a scalar. Fail.");
        return;
      }
      
      if (mxGetM(MASK_PARAMETER_IN) != 1 || mxGetN(MASK_PARAMETER_IN) != 1) {
        mexErrMsgTxt("Start must be a scalar. Fail.");
        return;
      }

      // fall through
      
    case 4:
      if (mxGetM(MASK_PARAMETER_IN) != 1 && mxGetN(MASK_PARAMETER_IN) != 1) {
        mexErrMsgTxt("Mask must be a row or column vector. Fail.");
        return;
      }  
        
      // fall through
      
    case 3:
      if (! mxIsEmpty(DIAG_VECTOR_PARAMETER_IN)) {
       if (mxIsSparse(DIAG_VECTOR_PARAMETER_IN)) {
          mexErrMsgTxt("Scale must be dense. Fail.");
          return;
        }
      
        if (! mxIsSingle(DIAG_VECTOR_PARAMETER_IN) &&
            ! mxIsDouble(DIAG_VECTOR_PARAMETER_IN)) {
          mexErrMsgTxt("Scale must be double or single. Fail.");
          return;
        }
      
        if (mxGetM(DIAG_VECTOR_PARAMETER_IN) != 1 ||
            mxGetN(DIAG_VECTOR_PARAMETER_IN) != mxGetN(SPARSE_MATRIX_PARAMETER_IN)) {
          mexErrMsgTxt("Scale has incompatible shape. Fail.");
          return;
        }
      }

      // fall through

    case 2:
      if (! mxIsSparse(SPARSE_MATRIX_PARAMETER_IN) || mxIsSparse(DENSE_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Require one sparse and one dense argument. Fail.");
        return;
      }
      if (mxGetM(SPARSE_MATRIX_PARAMETER_IN) != mxGetN(DENSE_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Arguments have incompatible shape. Fail.");
        return;
      }

      if (! mxIsSingle(DENSE_MATRIX_PARAMETER_IN) &&
          ! mxIsDouble(DENSE_MATRIX_PARAMETER_IN)) {
        mexErrMsgTxt("Dense argument must be double or single. Fail.");
        return;
      }

      break;
    default:
      mexErrMsgTxt("Wrong number of arguments. Fail.");
      return;
  }

  size_t Bcol = mxGetM(DENSE_MATRIX_PARAMETER_IN);

  // determine number of output rows
  
  size_t Arow = mxGetN(SPARSE_MATRIX_PARAMETER_IN);
  size_t nout = Arow;
  
  switch (nrhs) {
    case 5: { // ugh, legacy syntax
      size_t end = mxGetScalar(END_PARAMETER_IN);
      size_t start = mxGetScalar(MASK_PARAMETER_IN);
    
      if (end > Arow) {
        mexErrMsgTxt("End is too big.  Fail.");
        return;
      }
        
      if (start < 1) {
        mexErrMsgTxt("Start is too small.  Fail.");
        return;
      }
    
      nout = 1+end-start;
    }
      
      break;
    case 4:
      if (! mxIsDouble(MASK_PARAMETER_IN)) {
        mexErrMsgTxt("Mask must be double (for now). Fail.");
        return;
      }
      
      // TODO: bounds checking (?)

      nout = mxGetNumberOfElements(MASK_PARAMETER_IN);

      break;
    default:
      break;
  }
  
  mxClassID outputClass = mxIsSingle(DENSE_MATRIX_PARAMETER_IN) ? mxSINGLE_CLASS : mxDOUBLE_CLASS;
  plhs[0] = mxCreateNumericMatrix(Bcol, nout, outputClass, mxREAL);

  std::thread t[NUM_THREADS];
  size_t quot = nout/NUM_THREADS;

  for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i] = std::thread(sparsetic_times_densetic,
                       nrhs,
                       prhs,
                       plhs,
                       i * quot,
                       (i + 1) * quot);
  }

  sparsetic_times_densetic (nrhs, prhs, plhs, (NUM_THREADS - 1) * quot, nout);

  for (int i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i].join ();
  }

  return;
}
