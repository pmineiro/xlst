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
 * mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=4 -DUSE_MY_ERF treeroute.cpp
 * 
 * == linux ==
 * 
 * mex OPTIMFLAGS="-O3" -largeArrayDims -DNUM_THREADS=4 -v CXXFLAGS='$CXXFLAGS -std=c++0x -fPIC' treeroute.cpp
 */


#define RES_PARAMETER_IN       prhs[0]
#define DESIGN_MATRIX_IN       prhs[1]
#define ISRANDOM_PARAMETER_IN  prhs[2]
#define WHERE_VECTOR_OUT       plhs[0]

// ugh ... missing on windows (?)
#ifndef USE_MY_ERF
using std::erf;
#else
template<typename scalar>
scalar erf(scalar x)
{
    // constants
    scalar a1 =  0.254829592;
    scalar a2 = -0.284496736;
    scalar a3 =  1.421413741;
    scalar a4 = -1.453152027;
    scalar a5 =  1.061405429;
    scalar p  =  0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x);

    // A&S formula 7.1.26
    scalar t = 1.0/(1.0 + p*x);
    scalar y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return sign*y;
}
#endif

// ugh ... r2013 wants an ancient version of C++ ABI on linux
#if __GNUC__ == 4 && __GNUC_MINOR__ == 4
namespace std
{
  struct default_random_engine
    {
      unsigned short xsubi[3];

      default_random_engine (int seed) {
        xsubi[0] = (seed >> 16) & 0xffff;
        xsubi[1] = (seed >> 8) & 0xffff;
        xsubi[2] = seed & 0xffff;
      }
    };

  template<typename scalar>
  struct uniform_real_distribution
    {
      scalar l;
      scalar u;

      uniform_real_distribution (scalar _l, scalar _u) : l (_l), u (_u) { }

      scalar operator() (default_random_engine& g)
        {
          return l + (u - l) * erand48 (g.xsubi);
        }
    };
}
#endif

// X = D*S => X' = S'*D'

template<typename scalar>
static void
denseVec_times_sparse (scalar*        denseVec,
                       const mxArray* sparsetic,
                       size_t         start,
                       size_t         end,
                       const bool*    mask,
                       scalar*        Xtic)
{
  mwIndex* ir = mxGetIr (sparsetic);  /* Row indexing      */
  mwIndex* jc = mxGetJc (sparsetic);  /* Column count      */
  double* s  = mxGetPr (sparsetic);   /* Non-zero elements */

  for (size_t i = 0; i + start < end; ++i) {       /* Loop through rows of S */
    if (mask[i]) {                                 /* Skip masked rows */
      size_t ii = i+start;
      *Xtic = 0;
      for (mwIndex k=jc[ii]; k<jc[ii+1]; ++k) {    /* Loop through non-zeros columns of S */
        *Xtic += s[k] * denseVec[ir[k]];
      }
      ++Xtic;
    }
  }
}

template<typename scalar,
         typename generator,
         typename distribution>
static void
route2 (const mxArray*                 res,
        const mxArray*                 xtic,
        const std::unique_ptr<bool[]>& mask,
        size_t                         n_mask,
        size_t                         start,
        size_t                         end,
        double*                        where,
        bool                           israndom,
        generator&                     g,
        distribution&                  d,
        scalar*                        thisdp)
{
  if (n_mask) {
    mxArray* left = mxGetField (res, 0, "left");

    if (left) {
      mxArray* w = mxGetField (res, 0, "wtic");
      scalar b = static_cast<scalar> (mxGetScalar (mxGetField (res, 0, "b")));
      std::unique_ptr<bool[]> submask (new bool[end-start]);
      std::fill (submask.get (), submask.get () + end - start, false);
      size_t n_submask = 0;
      
      denseVec_times_sparse ((scalar*) mxGetData(w), xtic, start, end, mask.get (), thisdp);

      if (israndom) {
        scalar sigma = static_cast<scalar> (mxGetScalar (mxGetField (res, 0, "sigma")));

        for (size_t i = 0, j = 0; i + start < end && j < n_mask; ++i) {
          if (mask[i]) {
            thisdp[j] = scalar(0.5) + scalar(0.5) * erf ((thisdp[j] - b) / sigma);
            if (d (g) < thisdp[j]) {
              submask[i] = 1;
              ++n_submask;
            }
            ++j;
          }
        }
      }
      else {
        for (size_t i = 0, j = 0; i + start < end && j < n_mask; ++i) {
          if (mask[i]) {
            if (thisdp[j] > b) {
              submask[i] = 1;
              ++n_submask;
            }
            ++j;
          }
        }
      }
      
      // left recursion
      route2<scalar> (mxGetField (res, 0, "left"), xtic, submask, n_submask, 
                      start, end, where, israndom, g, d, thisdp);
      
      n_submask = 0;
      for (size_t i = 0; i + start < end; ++i) {
        if (mask[i]) {
          if (! submask[i]) ++n_submask;
          submask[i] = ! submask[i];
        }
      }

      // right recursion
      route2<scalar> (mxGetField (res, 0, "right"), xtic, submask, n_submask, 
                      start, end, where, israndom, g, d, thisdp);
    }
    else { // ! left
      double nodeid = mxGetScalar (mxGetField (res, 0, "nodeid"));
      for (size_t i = 0; i + start < end; ++i) {
        if (mask[i]) {
          where[i+start] = nodeid;
        }
      }
    }
  }
}

template<typename scalar>
static void
route (const mxArray* res,
       const mxArray* xtic,
       bool           israndom,
       double*        where,
       size_t         start,
       size_t         end,
       scalar*        thisdp)
{
  std::hash<std::thread::id> hasher;
  unsigned int preseed = static_cast<unsigned int> (std::chrono::system_clock::now().time_since_epoch().count());
#if __GNUC__ == 4 && __GNUC_MINOR__ == 4
  unsigned int stack_is_here;
  unsigned int moreseed = * reinterpret_cast<unsigned int*> (& stack_is_here);
#else
  unsigned int moreseed = static_cast<unsigned int> (hasher (std::this_thread::get_id ()));
#endif
  // std::mt19937 g (preseed + moreseed);
  std::default_random_engine g (preseed + moreseed);
  std::uniform_real_distribution<double> d (0.0, 1.0);
  std::unique_ptr<bool[]> mask (new bool[end-start]);
  std::fill (mask.get (), mask.get () + end - start, true);

  route2<scalar> (res, xtic, mask, end - start, start, end, where, israndom, g, d, thisdp);
}

template<typename scalar>
static void
preroute (const mxArray* prhs[],
          mxArray*       plhs[],
          size_t         start,
          size_t         end,
          scalar*        thisdp)
{
  route (RES_PARAMETER_IN,
         DESIGN_MATRIX_IN,
         mxIsLogicalScalarTrue (ISRANDOM_PARAMETER_IN),
         (double*) mxGetData (WHERE_VECTOR_OUT),
         start,
         end,
         thisdp);
}

static int first = 1;

void mexFunction( int nlhs, mxArray *plhs[], 
                  int nrhs, const mxArray*prhs[] ) { 
  if (first) {
    mexPrintf("treeroute using NUM_THREADS=%u\n",NUM_THREADS);
    mexEvalString("drawnow;");
    first=0;
  }

  if (nrhs != 3) {
      mexErrMsgTxt("Wrong number of arguments. Fail.");
      return;
  }

  if (nlhs != 1) {
      mexErrMsgTxt("Wrong number of outputs. Fail.");
      return;
  }

  if (! mxIsStruct (RES_PARAMETER_IN) ||
      ! mxGetField (RES_PARAMETER_IN, 0, "nodeid")) {
    mexErrMsgTxt("First argument must be tree node. Fail.");
    return;
  }

  if (! mxIsSparse (DESIGN_MATRIX_IN)) {
    mexErrMsgTxt("Second argument must be sparse. Fail.");
    return;
  }

  if (mxGetField (RES_PARAMETER_IN, 0, "wtic") && 
      mxGetM (DESIGN_MATRIX_IN) != 
      mxGetNumberOfElements (mxGetField (RES_PARAMETER_IN, 0, "wtic"))) {
    mexErrMsgTxt("Incompatible shape between design matrix and root node. Fail.");
    return;
  }

  if (mxGetM (ISRANDOM_PARAMETER_IN) != 1 || 
      mxGetN (ISRANDOM_PARAMETER_IN) != 1 || 
      ! mxIsLogical(ISRANDOM_PARAMETER_IN)) {
    mexErrMsgTxt("Third argument must be logical scalar. Fail.");
    return;
  }
 
  mwSize nout = mxGetN (DESIGN_MATRIX_IN);
  
  void* thisdp[NUM_THREADS];
  bool issingle = ! mxGetField(RES_PARAMETER_IN, 0, "wtic") || 
                  mxIsSingle (mxGetField (RES_PARAMETER_IN, 0, "wtic"));
  size_t scalarsize = issingle ? sizeof (float) : sizeof (double);
  
  for (size_t i = 0; i < NUM_THREADS; ++i) {
    thisdp[i] = mxMalloc (nout * scalarsize);
  }

  WHERE_VECTOR_OUT = mxCreateNumericMatrix(nout, 1, mxDOUBLE_CLASS, mxREAL);

  std::thread t[NUM_THREADS];
  size_t quot = nout/NUM_THREADS;

  for (size_t i = 0; i + 1 < NUM_THREADS; ++i) {
    if (issingle) {
      t[i] = std::thread(preroute<float>,
                         prhs,
                         plhs,
                         i * quot,
                         (i + 1) * quot,
                         (float*) thisdp[i]);
    }
    else { 
       t[i] = std::thread(preroute<double>,
                          prhs,
                          plhs,
                          i * quot,
                          (i + 1) * quot,
                          (double*) thisdp[i]);
    }       
  }

  if (issingle) {
    preroute (prhs, plhs, (NUM_THREADS - 1) * quot, nout, (float*) thisdp[NUM_THREADS-1]);
  }
  else {
    preroute (prhs, plhs, (NUM_THREADS - 1) * quot, nout, (double*) thisdp[NUM_THREADS-1]);
  }
  
  for (int i = 0; i + 1 < NUM_THREADS; ++i) {
    t[i].join ();
    mxFree (thisdp[i]);
  }

  return;
}
