MATLAB?=matlab

ifeq (,$(wildcard /proc/cpuinfo))
NUM_THREADS?=8
all: $(patsubst %.cpp,%.mexw64,$(wildcard *.cpp))
else
NUM_THREADS?=$(shell grep 'core id' /proc/cpuinfo | sort -u | wc -l)
all: $(patsubst %.cpp,%.mexa64,$(wildcard *.cpp))
endif

clean:
	rm -f $(wildcard *.mexa64) $(wildcard *.mexw64)

%.mexa64: %.cpp
	 ${MATLAB} -nodisplay -nojvm -r 'mex OPTIMFLAGS="-O3" -largeArrayDims -DNUM_THREADS=$(NUM_THREADS) -v CXXFLAGS='"'"'$$CXXFLAGS -std=c++0x -fPIC'"'"' $*.cpp -lmwblas; exit'

%.mexw64: %.cpp
	 ${MATLAB} -nodisplay -nojvm -r 'mex OPTIMFLAGS="/O2" -largeArrayDims -DNUM_THREADS=$(NUM_THREADS) -DUSE_MY_ERF $*.cpp -lmwblas; exit'
