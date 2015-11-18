#include "mex.h"
#include "matrix.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <thread>

/*
 * compilation, at matlab prompt: (adjust NUM_THREADS as appropriate)
 * 
 * == windows ==
 * 
 * mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=4 treepredict.cpp
 * 
 * == linux ==
 * 
 * mex OPTIMFLAGS="-O3" -largeArrayDims -DNUM_THREADS=4 -v CXXFLAGS='$CXXFLAGS -std=c++0x -fPIC' treepredict.cpp
 */


#define XTIC_MATRIX_IN         prhs[0]
#define OAS_MATRIX_IN          prhs[1]
#define FILTMAT_MATRIX_IN      prhs[2]
#define EXINDEX_MATRIX_IN      prhs[3]
#define BIAS_CELL_IN           prhs[4]
#define RES_PER_EX_SCALAR_IN   prhs[5]
#define Y_VECTOR_OUT           plhs[0]

template<unsigned int N>
static void
predict_one (const mxArray*     xtic,
             const mxArray*     oas,
             const mxArray*     filtmat,
             mwIndex            node,
             mwIndex            example,
             const float*       mybias,
             double*            yptr)
{
  const mwIndex* fmir = mxGetIr (filtmat);
  const mwIndex* fmjc = mxGetJc (filtmat);
  const mwIndex* xir = mxGetIr (xtic);
  const mwIndex* xjc = mxGetJc (xtic);
  const double* xs = mxGetPr (xtic);
  mwSize nexamples = mxGetN (xtic);
  mwSize nfeatures = mxGetM (oas);
  mwSize ncandidates = mxGetN (oas);
  const float* oasptr = (const float*) mxGetPr (oas);
  mwIndex argmax[N]; std::fill (argmax, argmax + N, mwIndex (0));
  float max[N]; std::fill (max, max + N, -FLT_MAX);

  for (mwIndex fmk = fmjc[node]; fmk < fmjc[node+1]; ++fmk) { 
    mwIndex candidate = fmir[fmk];
    const float* oascandidate = oasptr + (nfeatures * candidate);
    float pred = 0;

    for (mwIndex hk = xjc[example]; hk < xjc[example+1]; ++hk) {
      mwIndex feature = xir[hk];

      pred += oascandidate[feature] * xs[hk];
    }

    if (mybias) {
      pred += mybias[fmk - fmjc[node]];
    }

    mwIndex arg = 1 + candidate;

    for (unsigned int n = 0; n < N; ++n) {
      if (pred > max[n]) {
        std::swap (argmax[n], arg);
        std::swap (max[n], pred);
      }
    }
  }

  for (unsigned int n = 0; n < N; ++n) { 
    yptr[example + n * nexamples] = argmax[n];
  }
}

template<unsigned int N>
static void
predict (const mxArray* prhs[],
         double*        yptr,   
         mwIndex        offset,
         mwIndex        num_threads)
{
  const mwIndex* exir = mxGetIr (EXINDEX_MATRIX_IN);
  const mwIndex* exjc = mxGetJc (EXINDEX_MATRIX_IN);
  mwSize nnodes = mxGetN (EXINDEX_MATRIX_IN);

  for (mwIndex node = offset; node < nnodes; node += num_threads) { /* loop nodes */
    const mxArray* mybias = mxGetCell (BIAS_CELL_IN, node);
    const float* mybiasptr = (mybias) ? (const float*) mxGetData (mybias) : 0;
    for (mwIndex k = exjc[node]; k < exjc[node+1]; ++k) {           /* loop examples */
      mwIndex example = exir[k];
      predict_one<N> (XTIC_MATRIX_IN, OAS_MATRIX_IN, FILTMAT_MATRIX_IN, 
                      node, example, mybiasptr, yptr);
    }
  }
}

static int first = 1;

void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray*prhs[] ) { 
  if (first) {
    mexPrintf("treepredict using NUM_THREADS=%u\n",NUM_THREADS);
    mexEvalString("drawnow;");
    first=0;
  }

  if (nrhs != 6) {
      mexErrMsgTxt("Wrong number of arguments. Fail.");
      return;
  }

  if (nlhs != 1) {
      mexErrMsgTxt("Wrong number of outputs. Fail.");
      return;
  }

  if (! mxIsSparse (XTIC_MATRIX_IN)) {
    mexErrMsgTxt("First argument must be sparse. Fail.");
    return;
  }

  if (mxIsSparse (OAS_MATRIX_IN)) {      
    mexErrMsgTxt("Second argument must be dense. Fail.");
    return;
  }

  if (! mxIsSingle (OAS_MATRIX_IN)) {
    mexErrMsgTxt("Second argument must be single. Fail.");
    return;
  }

  if (mxGetM (OAS_MATRIX_IN) != mxGetM (XTIC_MATRIX_IN)) {
    mexErrMsgTxt("First and second argument have incompatible shape. Fail.");
    return;
  }

  if (! mxIsSparse (FILTMAT_MATRIX_IN)) { 
    mexErrMsgTxt("Third argument must be sparse. Fail.");
    return;
  }

  if (mxGetN (OAS_MATRIX_IN) != mxGetM (FILTMAT_MATRIX_IN)) {
    mexErrMsgTxt("Second and third argument have incompatible shape. Fail.");
    return;
  }

  if (! mxIsSparse (EXINDEX_MATRIX_IN)) { 
    mexErrMsgTxt("Fourth argument must be sparse. Fail.");
    return;
  }

  if (mxGetM (EXINDEX_MATRIX_IN) != mxGetN (XTIC_MATRIX_IN)) {
    mexErrMsgTxt("First and fourth argument have incompatible shape. Fail.");
    return;
  }

  if (mxGetN (EXINDEX_MATRIX_IN) != mxGetN (FILTMAT_MATRIX_IN)) {
    mexErrMsgTxt("Third and fourth argument have incompatible shape. Fail.");
    return;
  }

  if (! mxIsCell (BIAS_CELL_IN)) {
    mexErrMsgTxt("Fifth argument must be cell array. Fail.");
    return;
  }

  for (mwIndex i = 0; i < mxGetNumberOfElements (BIAS_CELL_IN); ++i) {
    const mxArray* mybias = mxGetCell (BIAS_CELL_IN, i);

    if (mybias) {
      if (! mxIsSingle (mybias)) {
        mexErrMsgTxt("Fifth argument must contain single arrays. Fail.");
      }

      break;
    }
  }

  if (! mxIsDouble (RES_PER_EX_SCALAR_IN) || mxGetM (RES_PER_EX_SCALAR_IN) != 1 ||
      mxGetN (RES_PER_EX_SCALAR_IN) != 1) {
    mexErrMsgTxt("Sixth argument must be a double scalar. Fail.");
  }

  if (mxGetScalar (RES_PER_EX_SCALAR_IN) != 1 && mxGetScalar (RES_PER_EX_SCALAR_IN) != 5) { 
    mexErrMsgTxt("Sixth argument must be either 1 or 5. Fail.");
  }

  mwSize nout = mxGetM (EXINDEX_MATRIX_IN);
  mwSize resperex = (mwSize) mxGetScalar (RES_PER_EX_SCALAR_IN);
  Y_VECTOR_OUT = mxCreateNumericMatrix (nout, resperex, mxDOUBLE_CLASS, mxREAL);
  double* yptr = (double*) mxGetPr (Y_VECTOR_OUT);

  std::thread t[NUM_THREADS];

  if (mxGetScalar (RES_PER_EX_SCALAR_IN) == 5) {
    for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
      t[i] = std::thread (predict<5>, prhs, yptr, i, NUM_THREADS);
    }
  
    predict<5> (prhs, yptr, NUM_THREADS-1, NUM_THREADS);
  }
  else {
    for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
      t[i] = std::thread (predict<1>, prhs, yptr, i, NUM_THREADS);
    }
  
    predict<1> (prhs, yptr, NUM_THREADS-1, NUM_THREADS);
  }
  
  for (int i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i].join ();
  }

  return;
}
