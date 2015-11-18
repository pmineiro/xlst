#include "mex.h"
#include "matrix.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <thread>

/*
 * compilation, at matlab prompt: (adjust NUM_THREADS as appropriate)
 * 
 * == windows ==
 * 
 * mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=4 -DUSE_MY_ERF treemakeimpweights.cpp
 * 
 * == linux ==
 * 
 * mex OPTIMFLAGS="-O3" -largeArrayDims -DNUM_THREADS=4 -v CXXFLAGS='$CXXFLAGS -std=c++0x -fPIC' treemakeimpweights.cpp
 */


#define YTIC_MATRIX_IN          prhs[0]
#define FILTMAT_MATRIX_IN       prhs[1]
#define RX_VECTOR_IN            prhs[2]
#define IMPWEIGHTS_VECTOR_OUT   plhs[0]
#define AVGSOME_SCALAR_OUT      plhs[1]
#define MAXSOME_SCALAR_OUT      plhs[2]

// impweights=full(dot(ytic,filtmat(:,rx),1));
// avgsome=avgsome+full(sum(sum(filtmat(:,rx))));
// maxsome=full(max(sum(filtmat(:,rx))));

// X = D*S => X' = S'*D'

static void
make_impweights_impl (const mxArray* ytic,
                      const mxArray* filtmat,
                      const double*  rx,
                      size_t         start,
                      size_t         end,
                      double*        impweights,
                      double*        avgsome,
                      double*        maxsome)
{
  mwIndex* yir = mxGetIr (ytic);  /* Row indexing      */
  mwIndex* yjc = mxGetJc (ytic);  /* Column count      */
  double* ys  = mxGetPr (ytic);   /* Non-zero elements */
  mwIndex* fmir = mxGetIr (filtmat);  /* Row indexing      */
  mwIndex* fmjc = mxGetJc (filtmat);  /* Column count      */
  double* fms  = mxGetPr (filtmat);   /* Non-zero elements */
  mwSize fmcols = mxGetN (filtmat);

  for (size_t i = start; i < end; ++i) {       /* Loop through rows of S */
    size_t rxi = static_cast<size_t> (rx[i]);
    
    if (rxi > 0 && rxi <= fmcols) {
      double dot = 0;
      double thissum = 0;
    
      mwIndex yk = yjc[i];
      mwIndex fmk = fmjc[rxi-1];

      mwIndex yi = yir[yk];
      mwIndex fmi = fmir[fmk];

      while (yk < yjc[i+1] && fmk < fmjc[rxi]) {
        if (yi == fmi) {
          dot += ys[yk] * fms[fmk];
          thissum += fms[fmk];
          yi = yir[++yk];
          fmi = fmir[++fmk];
        }
        else if (yi < fmi) { 
          yi = yir[++yk];
        }
        else {
          thissum += fms[fmk];
          fmi = fmir[++fmk];
        }
      }
      
      while (fmk < fmjc[rxi]) {
        thissum += fms[fmk];
        ++fmk;
      }
      
      impweights[i] = dot;
     
      *avgsome += thissum;
      *maxsome = std::max<double> (*maxsome, thissum);
    }
  }
}

static void
make_impweights (const mxArray* prhs[], 
                 mxArray*       plhs[], 
                 size_t         start,
                 size_t         end,
                 double*        scratch)
{
  make_impweights_impl (YTIC_MATRIX_IN,
                        FILTMAT_MATRIX_IN,
                        (const double*) mxGetData (RX_VECTOR_IN),
                        start,
                        end,
                        (double*) mxGetData (IMPWEIGHTS_VECTOR_OUT),
                        scratch,
                        scratch + 1);
}

static int first = 1;

void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray*prhs[] ) { 
  if (first) {
    mexPrintf("treemakeimpweights using NUM_THREADS=%u\n",NUM_THREADS);
    mexEvalString("drawnow;");
    first=0;
  }

  if (nrhs != 3) {
      mexErrMsgTxt("Wrong number of arguments. Fail.");
      return;
  }

  if (nlhs < 1 || nlhs > 3) {
      mexErrMsgTxt("Wrong number of outputs. Fail.");
      return;
  }

  if (! mxIsSparse (YTIC_MATRIX_IN)) {
    mexErrMsgTxt("First argument must be sparse. Fail.");
    return;
  }

  if (! mxIsSparse (FILTMAT_MATRIX_IN)) {
    mexErrMsgTxt("Second argument must be sparse. Fail.");
    return;
  }

  if (mxGetM (YTIC_MATRIX_IN) != mxGetM (FILTMAT_MATRIX_IN)) {
    mexErrMsgTxt("First two arguments have incompatible shapes. Fail.");
    return;
  }

  if (! mxIsDouble (RX_VECTOR_IN)) {
    mexErrMsgTxt("Third argument must be double. Fail.");
    return;
  }

  if (mxGetNumberOfElements (RX_VECTOR_IN) != mxGetN (YTIC_MATRIX_IN)) {
    mexErrMsgTxt("First and third arguments have incompatible shapes. Fail.");
    return;
  }
 
  mwSize nout = mxGetNumberOfElements (RX_VECTOR_IN);

  IMPWEIGHTS_VECTOR_OUT = mxCreateNumericMatrix (1, nout, mxDOUBLE_CLASS, mxREAL);
  double* scratch = (double *) mxCalloc (2 * NUM_THREADS, sizeof (double));
  
  std::thread t[NUM_THREADS];
  size_t quot = nout/NUM_THREADS;

  for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i] = std::thread (make_impweights,
                        prhs,
                        plhs,
                        i * quot,
                        (i + 1) * quot,
                        scratch + 2 * i);
  }

  make_impweights (prhs, plhs, (NUM_THREADS - 1) * quot, nout, scratch + 2*(NUM_THREADS-1));

  double avgsome = scratch[2*(NUM_THREADS-1)];
  double maxsome = scratch[2*(NUM_THREADS-1)+1];
  
  for (int i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i].join ();
    avgsome += scratch[2*i];
    maxsome = std::max (maxsome, scratch[2*i+1]);
  }

  switch (nlhs)
    {
      case 3:
        MAXSOME_SCALAR_OUT = mxCreateNumericMatrix (1, 1, mxDOUBLE_CLASS, mxREAL);
        ((double*) mxGetData (MAXSOME_SCALAR_OUT))[0] = maxsome;
        
        // fall through
      case 2:
        AVGSOME_SCALAR_OUT = mxCreateNumericMatrix (1, 1, mxDOUBLE_CLASS, mxREAL);
        ((double*) mxGetData (AVGSOME_SCALAR_OUT))[0] = avgsome / nout;
        // fall through
      default:
        break;
    }

  mxFree (scratch);

  return;
}
