# Parallel matrix factorization for recommender systems
## Motivation
We chose to implement the paper "Parallel matrix factorizaiton for recommender systems" by Yu et. al. 2013 due to its real life applications in every day use like product suggestions, and its large parallelizability.

## Implementation
Our implementation began with a serial implementation where we built CCDPP.cpp, MatrixFactorization.cpp, Rating.h, RatingLoader.cpp and sparseRatings.cpp. This layed the ground work for which we could build parallel and optimized implementations off of.

We began by writing an un-optimized parallel version of the paper's parallel CCD++ algorithm in ParallelCCDPP.cpp. We utilized OpenMP and "for schedule(dynamic)" along with a few other small tnes to achieve a correct parallel implementation which was tested on random matrices and a dataset suggested in the paper.

We built optimizations off of the parallel version of the CCD++ algorithm. This included adaptive stopping, and restricted thread count. These optimizations can be found in OptParallelCCDPP.cpp.
The other optimization: iteration count, was added to MatrixFactorization.cpp, which acts as the pilot for our code.

## Running the code
Our suggestion for running the code is to open a linux terminal, or wsl terminal in the root directory: MatrixFactorization. From this directory, you should be able to run Make which will execute the Makefile and build binaries.
The main binary is ./ccdpp and all implementations can be accessed from this binary. This binary accepts multiple arguments/flags. The full argument field is

./ccdpp intput_filename -option

examples for option are

-s (serial implementation)
-p (parallel implementation)
-o (optimized implementation)

the optimized implementation also accepts arguments -n numthreads and -e epsilon
For example, running the optimized implementation on file test.csv with 4 threads and epsilon(adaptive stopping) value of 0.002 would look like

./ccdpp test.csv -o -n 4 -e 0.002

additionally, if no input file is given, it will run on a small test matrix.