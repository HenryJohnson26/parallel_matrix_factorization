CXX := g++
CXXFLAGS := -std=c++17 -O2

SRCS := CCDPP.cpp ParallelCCDPP.cpp MatrixFactorization.cpp SparseRatings.cpp RatingLoader.cpp
PAR_SRCS := $(SRCS) ParallelCCDPP.cpp

all: ccdpp ccdpp_parallel

ccdpp: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

ccdpp_parallel: $(PAR_SRCS)
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ $^

clean:
	rm -f ccdpp ccdpp_parallel