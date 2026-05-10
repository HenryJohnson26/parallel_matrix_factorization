CXX := g++
CXXFLAGS := -std=c++17 -O2 -fopenmp

SRCS := CCDPP.cpp MatrixFactorization.cpp SparseRatings.cpp RatingLoader.cpp ParallelCCDPP.cpp OptParallelCCDPP.cpp

ccdpp: $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f ccdpp
