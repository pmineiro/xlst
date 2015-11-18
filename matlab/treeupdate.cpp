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
 * mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=4 treeupdate.cpp
 * 
 * == linux ==
 * 
 * mex OPTIMFLAGS="-O3" -largeArrayDims -DNUM_THREADS=4 -v CXXFLAGS='$CXXFLAGS -std=c++0x -fPIC' treeupdate.cpp
 */


#define HASHXTIC_MATRIX_IN     prhs[0]
#define OAS_MATRIX_IN          prhs[1]
#define FILTMAT_MATRIX_IN      prhs[2]
#define EXINDEX_MATRIX_IN      prhs[3]
#define BIAS_CELL_IN           prhs[4]
#define MOMENTUM_MATRIX_IN     prhs[5]
#define MOMENTUM_BIAS_CELL_IN  prhs[6]
#define YTIC_MATRIX_IN         prhs[7]
#define C_VECTOR_IN            prhs[8]
#define NODE_PERM_VECTOR_IN    prhs[9]
#define EX_PERM_VECTOR_IN      prhs[10]
#define ETA_SCALAR_IN          prhs[11]
#define ALPHA_SCALAR_IN        prhs[12]
#define SOFTMAX_LOGICAL_IN     prhs[13]
#define NORM_DELTA_OUT         plhs[0]

template<bool softmax>
static float
update_one (const mxArray*      hashxtic,
            const mxArray*      oas,
            const mxArray*      filtmat,
            const mxArray*      momentum,
            mwIndex             node,
            mwIndex             example,
            float*              mybias,
            float*              mymombias,
            const mxArray*      ytic,
            const mxArray*      C,
            float               eta,
            float               alpha,
            mwIndex             nexamples)
{
  const mwIndex* fmir = mxGetIr (filtmat);
  const mwIndex* fmjc = mxGetJc (filtmat);
  const mwIndex* hashxir = mxGetIr (hashxtic);
  const mwIndex* hashxjc = mxGetJc (hashxtic);
  const double* hashxs = mxGetPr (hashxtic);
  mwSize nfeatures = mxGetM (oas);
  mwSize ncandidates = mxGetN (oas);
  float* oasptr = (float*) mxGetPr (oas);
  float* momptr = (float*) mxGetPr (momentum);
  const mwIndex* yir = mxGetIr (ytic);
  const mwIndex* yjc = mxGetJc (ytic);
  const double* ys = mxGetPr (ytic);
  const float* Cptr = (const float *) mxGetPr (C);
  float max = -FLT_MAX;
  
  // ... compute predictions ... //

  mwIndex candidates = fmjc[node+1]-fmjc[node];

  float* preds = (float*) alloca (candidates * sizeof (float));
  memset (preds, 0, candidates * sizeof (float));

  for (mwIndex fmk = fmjc[node]; fmk < fmjc[node+1]; ++fmk) { 
    mwIndex candidate = fmir[fmk];
    mwIndex c = fmk - fmjc[node];
    const float* oascandidate = oasptr + (nfeatures * candidate);

    for (mwIndex hk = hashxjc[example]; hk < hashxjc[example+1]; ++hk) {
      mwIndex feature = hashxir[hk];

      preds[c] += oascandidate[feature] * hashxs[hk];
    }

    preds[c] += mybias[c];

    if (preds[c] > max) {
      max = preds[c];
    }
  }

  if (softmax) {
    float sumpreds = 0;

    for (mwIndex c = 0; c < candidates; ++c) {
      preds[c] = exp (preds[c] - max);
      sumpreds += preds[c];
    }

    for (mwIndex c = 0; c < candidates; ++c) {
      preds[c] = preds[c] / sumpreds;
    }
  }
  else {
    for (mwIndex c = 0; c < candidates; ++c) {
      if (preds[c] > 0) {
        preds[c] = 1.0 / (1.0 + exp (-preds[c]));
      }
      else {
        float exppred = exp (preds[c]);
        preds[c] = exppred / (1.0 + exppred);
      }
    }
  }
  
  // ... subtract true labels ... //

  for (mwIndex fmk = fmjc[node], yk = yjc[example]; 
       fmk < fmjc[node+1] && yk < yjc[example+1]; ) {
    mwIndex candidate = fmir[fmk];
    mwIndex truecandidate = yir[yk];

    if (candidate == truecandidate) {
      mwIndex c = fmk - fmjc[node];
      preds[c] -= ys[yk];

      ++fmk;
      ++yk;
    }
    else if (candidate < truecandidate) {
      ++fmk;
    }
    else { 
      ++yk;
    }
  }
  
  // ... compute norm gradient ... //

  float norm_delta = 0;

  for (mwIndex c = 0; c < candidates; ++c) {
    norm_delta += preds[c] * preds[c];
  }
  
  // ... update model ... //

  for (mwIndex fmk = fmjc[node]; fmk < fmjc[node+1]; ++fmk) { 
    mwIndex candidate = fmir[fmk];
    mwIndex c = fmk - fmjc[node];
    float* momcandidate = momptr + (nfeatures * candidate);
    float* oascandidate = oasptr + (nfeatures * candidate);

    for (mwIndex hk = hashxjc[example]; hk < hashxjc[example+1]; ++hk) {
      mwIndex feature = hashxir[hk];
      float g = hashxs[hk] * preds[c] / Cptr[feature];

      momcandidate[feature] = alpha * momcandidate[feature] - eta * g;
      oascandidate[feature] += momcandidate[feature];
    }

    mymombias[c] = alpha * mymombias[c] - (eta / nexamples) * preds[c];
    mybias[c] += mymombias[c];
  }

  return norm_delta;
}

template<bool softmax>
static void
update (const mxArray* prhs[],
        double*        norm_deltas,
        mwIndex        offset,
        mwIndex        num_threads)
{
  const mwIndex* exir = mxGetIr (EXINDEX_MATRIX_IN);
  const mwIndex* exjc = mxGetJc (EXINDEX_MATRIX_IN);
  mwSize nnodes = mxGetN (EXINDEX_MATRIX_IN);
  const double* nodeperm = (const double*) mxGetPr (NODE_PERM_VECTOR_IN);
  const double* experm = (const double*) mxGetPr (EX_PERM_VECTOR_IN);

  float eta = mxGetScalar (ETA_SCALAR_IN);
  float alpha = mxGetScalar (ALPHA_SCALAR_IN);

  *norm_deltas = 0;

  for (mwIndex ii = offset; ii < nnodes; ii += num_threads) { /* loop nodes */
    mwIndex node = nodeperm[ii] - 1;
    const mxArray* mybias = mxGetCell (BIAS_CELL_IN, node);
    const mxArray* mymombias = mxGetCell (MOMENTUM_BIAS_CELL_IN, node);
    if (mybias && mymombias) { /* should always be true, but just in case ... */
      float* mybiasptr = (float*) mxGetData (mybias);
      float* mymombiasptr = (float*) mxGetData (mymombias);
      mwIndex nexamples = exjc[node+1]-exjc[node];
      for (mwIndex k = exjc[node]; k < exjc[node+1]; ++k) {     /* loop examples */
         mwIndex example = experm[exir[k]] - 1;
         *norm_deltas += 
           update_one<softmax> (HASHXTIC_MATRIX_IN, OAS_MATRIX_IN, 
                                FILTMAT_MATRIX_IN, MOMENTUM_MATRIX_IN, 
                                node, example, mybiasptr, mymombiasptr, 
                                YTIC_MATRIX_IN, C_VECTOR_IN, 
                                eta, alpha, nexamples);
      }
    }
  }
}

static int first = 1;

void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray*prhs[] ) { 
  if (first) {
    mexPrintf("treeupdate using NUM_THREADS=%u\n",NUM_THREADS);
    mexEvalString("drawnow;");
    first=0;
  }

  if (nrhs != 14) {
      mexErrMsgTxt("Wrong number of arguments. Fail.");
      return;
  }

  if (nlhs != 1) {
      mexErrMsgTxt("Wrong number of outputs. Fail.");
      return;
  }

  if (! mxIsSparse (HASHXTIC_MATRIX_IN)) {
    mexErrMsgTxt("Hashxtic must be sparse. Fail.");
    return;
  }

  if (mxIsSparse (OAS_MATRIX_IN)) {      
    mexErrMsgTxt("Oas must be dense. Fail.");
    return;
  }

  if (! mxIsSingle (OAS_MATRIX_IN)) {
    mexErrMsgTxt("Oas must be single. Fail.");
    return;
  }

  if (mxGetM (OAS_MATRIX_IN) != mxGetM (HASHXTIC_MATRIX_IN)) {
    mexErrMsgTxt("Hashxtic and oas have incompatible shape. Fail.");
    return;
  }

  if (! mxIsSparse (FILTMAT_MATRIX_IN)) { 
    mexErrMsgTxt("Filtmat must be sparse. Fail.");
    return;
  }

  if (mxGetN (OAS_MATRIX_IN) != mxGetM (FILTMAT_MATRIX_IN)) {
    mexErrMsgTxt("Oas and filtmat have incompatible shape. Fail.");
    return;
  }

  if (! mxIsSparse (EXINDEX_MATRIX_IN)) { 
    mexErrMsgTxt("Exindex must be sparse. Fail.");
    return;
  }

  if (mxGetN (EXINDEX_MATRIX_IN) != mxGetN (FILTMAT_MATRIX_IN)) {
    mexErrMsgTxt("Exindex and filtmat have incompatible shape. Fail.");
    return;
  }
  
  if (! mxIsCell (BIAS_CELL_IN)) {
    mexErrMsgTxt("Bias must be cell array. Fail.");
    return;
  }

  for (mwIndex i = 0; i < mxGetNumberOfElements (BIAS_CELL_IN); ++i)
    {
      const mxArray* mybias = mxGetCell (BIAS_CELL_IN, i);

      if (mybias) {
        if (! mxIsSingle (mybias)) {
          mexErrMsgTxt("Bias must contain single arrays. Fail.");
        }

        break;
      }
    }

  if (mxIsSparse (MOMENTUM_MATRIX_IN)) {      
    mexErrMsgTxt("Momentum must be dense. Fail.");
    return;
  }

  if (! mxIsSingle (MOMENTUM_MATRIX_IN)) {
    mexErrMsgTxt("Momentum must be single. Fail.");
    return;
  }

  if (! mxIsCell (MOMENTUM_BIAS_CELL_IN)) {
    mexErrMsgTxt("Momentum bias must be cell array. Fail.");
    return;
  }

  for (mwIndex i = 0; i < mxGetNumberOfElements (MOMENTUM_BIAS_CELL_IN); ++i)
    {
      const mxArray* mymombias = mxGetCell (MOMENTUM_BIAS_CELL_IN, i);

      if (mymombias) {
        if (! mxIsSingle (mymombias)) {
          mexErrMsgTxt("Momentum bias must contain single arrays. Fail.");
        }

        break;
      }
    }

  if (! mxIsSparse (YTIC_MATRIX_IN)) {
    mexErrMsgTxt("Ytic must be sparse. Fail.");
    return;
  }

  if (mxGetN (YTIC_MATRIX_IN) != mxGetN (HASHXTIC_MATRIX_IN)) {
    mexErrMsgTxt("Ytic and hashxtic have incompatible shapes. Fail.");
    return;
  }

  if (mxGetM (YTIC_MATRIX_IN) != mxGetM (FILTMAT_MATRIX_IN)) {
    mexErrMsgTxt("Ytic and filtmat have incompatible shapes. Fail.");
    return;
  }

  if (! mxIsSingle (C_VECTOR_IN)) {
    mexErrMsgTxt("C must be single. Fail.");
    return;
  }

  if (mxGetNumberOfElements (C_VECTOR_IN) != mxGetM (OAS_MATRIX_IN)) {
    mexErrMsgTxt("C and oas have incompatible shapes. Fail.");
    return;
  }
  
  if (! mxIsDouble (NODE_PERM_VECTOR_IN)) {
    mexErrMsgTxt("Node perm must be double. Fail.");
    return;
  }

  if (mxGetNumberOfElements (NODE_PERM_VECTOR_IN) != mxGetN (FILTMAT_MATRIX_IN)) {
    mexErrMsgTxt("Node perm and filtmat have incompatible shapes. Fail.");
    return;
  } 

  if (! mxIsDouble (EX_PERM_VECTOR_IN)) {
    mexErrMsgTxt("Example perm must be double. Fail.");
    return;
  }

  if (mxGetNumberOfElements (EX_PERM_VECTOR_IN) != mxGetN (HASHXTIC_MATRIX_IN)) {
    mexErrMsgTxt("Example perm and hashxtic have incompatible shapes. Fail.");
    return;
  } 


  if (mxGetM (ETA_SCALAR_IN) != 1 || mxGetN (ETA_SCALAR_IN) != 1) {
    mexErrMsgTxt("Eta must be a double scalar. Fail.");
    return;
  }

  if (mxGetScalar (ETA_SCALAR_IN) <= 0) {
    mexErrMsgTxt("Eta must be positive. Fail.");
    return;
  }

  if (mxGetM (ALPHA_SCALAR_IN) != 1 || mxGetN (ALPHA_SCALAR_IN) != 1) {
    mexErrMsgTxt("Alpha must be a double scalar. Fail.");
    return;
  }

  if (mxGetScalar (ALPHA_SCALAR_IN) < 0 || mxGetScalar (ALPHA_SCALAR_IN) >= 1) {
    mexErrMsgTxt("Alpha must be in [0,1). Fail.");
    return;
  }

  if (! mxIsLogical (SOFTMAX_LOGICAL_IN) || mxGetM (SOFTMAX_LOGICAL_IN) != 1
      || mxGetN (SOFTMAX_LOGICAL_IN) != 1) {
    mexErrMsgTxt("Softmax option must be logical. Fail.");
    return;
  }

  NORM_DELTA_OUT = mxCreateNumericMatrix (1, 1, mxDOUBLE_CLASS, mxREAL);
  double* norm_delta_out = (double *) mxGetPr (NORM_DELTA_OUT);
  double* norm_deltas = (double*) mxCalloc(NUM_THREADS, sizeof(double));

  std::thread t[NUM_THREADS];

  if (mxIsLogicalScalarTrue (SOFTMAX_LOGICAL_IN)) {
    for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
      t[i] = std::thread (update<true>, prhs, norm_deltas + i, i, NUM_THREADS);
    }

    update<true> (prhs, norm_deltas + NUM_THREADS-1, NUM_THREADS-1, NUM_THREADS);
  }
  else {
    for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
      t[i] = std::thread (update<false>, prhs, norm_deltas + i, i, NUM_THREADS);
    }

    update<false> (prhs, norm_deltas + NUM_THREADS-1, NUM_THREADS-1, NUM_THREADS);
  }
  
  for (int i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i].join ();
    norm_deltas[NUM_THREADS-1] += norm_deltas[i];
  }

  *norm_delta_out = norm_deltas[NUM_THREADS-1];

  return;
}
