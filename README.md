# Fast Bayesian Estimation Using Location-type Variational Representations
This is the official repository that hosts the code to the EUSIPCO 2025 paper
"Fast Bayesian Estimation Using Location-type Variational Representations". 
Preprint of the article is available in the repository root.

![Convergence of the Type I and the proposed method on a toy 2d
problem.](contour.png)

# How to use the code
Here `type_I.cpp` implements the reference method and `type_II.cpp` the proposed
method.

## Dependencies
Before compiling the code, install the following dependencies.
* [gls (Gnu Scientific Library)](https://www.gnu.org/software/gsl/)
* [libTiff](https://libtiff.gitlab.io/libtiff/)
* [sciplot](https://sciplot.github.io/)
* [OpenBLAS](http://www.openmathlib.org/OpenBLAS/)

## Compile & run
To compile and run the code do the following. After the program finishes two new
files will appear in the `build` directory. `example.pdf` plots the runtime
versus gamma parameter of both methods. `example.csv` contains the data used to
generate the plot.

```shell
# Compile the code
mkdir build
cd build
cmake ..
make

# Run it
./location_typeII
```
