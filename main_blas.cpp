#include <iostream>
#include <sstream>
#include <assert.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_randist.h>

#define cimg_use_tiff
#include "CImg.h"
#include <sciplot/sciplot.hpp>

#include "anscombe.h"
#include "utils.h"
#include "type_I.h"
#include "type_II.h"

using namespace cimg_library;
using namespace sciplot;

int generate_data(const gsl_rng *gen, const gsl_matrix *Lam, double gamma, gsl_vector *x, gsl_vector *y) {
//    J = Lam + np.outer(v, v)
//    Sigma = np.linalg.inv(J)
//    mu = Sigma @ v * gamma
//    x = np.random.multivariate_normal(mu, Sigma)
//    X  = x.reshape(rows, cols).astype(np.float32)

    // Generate ground-truth
    int N = Lam->size1;
    // Covariance matrix
    gsl_matrix *L = gsl_matrix_alloc(N, N);
    gsl_matrix_memcpy(L, Lam);
    gsl_matrix_add_constant(L, 1);
    gsl_linalg_cholesky_decomp1(L);
    gsl_linalg_cholesky_invert(L);
    // Mean vector
    gsl_vector *mu = gsl_vector_alloc(N);
    gsl_vector_set_all(x, gamma);
    gsl_blas_dgemv(CblasNoTrans, 1.0, L, x, 0.0, mu);
    // Sample
    gsl_linalg_cholesky_decomp1(L);
    gsl_ran_multivariate_gaussian(gen, mu, L, x);
    
    // Generate observations
    for (int n = 0; n < N; n++) {
        y->data[n] = gsl_ran_poisson(gen, x->data[n]);
    }
    gsl_matrix_free(L);
    gsl_vector_free(mu);
    return 0;
}

int main(int argc, char **argv) {
    size_t cols = 100;
    size_t rows = 100;
    size_t N = rows * cols;

    // Construct H matrix
    gsl_matrix *H = gsl_matrix_alloc(N, N);
    eye(0.0, H, 0);

    // Construct Lam matrix
    // Qx: matrix calculating the x-derivatives
    gsl_matrix *Sx = gsl_matrix_alloc(cols, cols);
    eye(0.0, Sx, -1);
    eye(-1.0, Sx, 0);
    gsl_matrix_const_view Sx_view = gsl_matrix_const_submatrix(Sx, 1, 0, cols-1, cols);
    gsl_matrix *Iy = gsl_matrix_alloc(rows, rows);
    eye(0.0, Iy, 0);
    gsl_matrix *Qx = gsl_matrix_alloc(rows*(cols-1), rows*cols);
    kronecker(Iy, &Sx_view.matrix, Qx);
    gsl_matrix_free(Sx);
    gsl_matrix_free(Iy);

    // Qy: matrix calculating the y-derivatives
    gsl_matrix *Sy = gsl_matrix_alloc(rows, rows);
    eye(0.0, Sy, -1);
    eye(-1.0, Sy, 0);
    gsl_matrix_const_view Sy_view = gsl_matrix_const_submatrix(Sy, 1, 0, rows-1, rows);
    gsl_matrix *Ix = gsl_matrix_alloc(cols, cols);
    eye(0.0, Ix, 0);
    gsl_matrix *Qy = gsl_matrix_alloc((rows-1)*cols, rows*cols);
    kronecker(&Sy_view.matrix, Ix, Qy);
    gsl_matrix_free(Sy);
    gsl_matrix_free(Ix);

    // Q = [Qx; Qy]
    gsl_matrix *Q = gsl_matrix_alloc(rows*(cols-1) + (rows-1)*cols, N);
    for (int n = 0; n < rows*(cols-1)*N; n++) {
        Q->data[n] = Qx->data[n];
    }
    for (int n = 0; n < (rows-1)*cols*N; n++) {
        Q->data[n + rows*(cols-1)*N] = Qy->data[n];
    }
    gsl_matrix *Lam = gsl_matrix_alloc(N, N);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, Q, Q, 0.0, Lam);
    gsl_matrix_free(Qx);
    gsl_matrix_free(Qy);
    gsl_matrix_free(Q);

    // Prepare random generator
    gsl_rng *gen = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(gen, 42);
    
    // Start the experiments
    gsl_vector *x = gsl_vector_alloc(N);
    gsl_vector *y = gsl_vector_alloc(N);
    gsl_vector *eta = gsl_vector_alloc(N);
    gsl_vector *x_hat = gsl_vector_alloc(N);
    const int n_iter = 600;     // Maximal number of iterations
    const int n_scales = 10;    // Number of different gammas
    const int n_real = 1;      // Number of realizations
    const double eps = 1e-5;    // Relative tolerance for stopping condition
    
    // Iterate through gammas
    std::vector<double> gamma(n_scales);
    std::vector<double> runtime_typeI(n_scales);
    std::vector<double> runtime_typeII(n_scales);
    std::vector<double> iters_typeI(n_scales);
    std::vector<double> iters_typeII(n_scales);
    clock_t start;
    double runtime;
    int iters;
    for (int scale = 0; scale < n_scales; scale++) {
        // Compute gamma
        gamma[scale] = scale + 1;
        
        // Iterate through realizations
        for (int real = 0; real < n_real; real++) {
            std::cout << "Scale: " << scale << ", realization: " << real << std::endl;
            // Generate data
            generate_data(gen, Lam, N*gamma[scale], x, y);
            
            // Calculate eta
            for (int n = 0; n < N; n++) {
                eta->data[n] = beta_prime2(0, y->data[n], 0.0);
            }

            // Run inference using Type I
            std::cout << "Type I" << std::endl;
            start = clock();
            iters = type_I(x_hat, y, H, Lam, eta, 0.0, n_iter, eps);
            runtime = ((double)(clock() - start)) / CLOCKS_PER_SEC;
            runtime_typeI[scale] = (real*runtime_typeI[scale] + runtime) / (real + 1);
            iters_typeI[scale] = (real*iters_typeI[scale] + iters) / (real + 1);

            // Run inference using Type II
            std::cout << "Type II" << std::endl;
            start = clock();
            iters = type_II(x_hat, y, H, Lam, eta, 0.0, n_iter, eps);
            runtime = ((double)(clock() - start)) / CLOCKS_PER_SEC;
            runtime_typeII[scale] = (real*runtime_typeII[scale] + runtime) / (real + 1);
            iters_typeII[scale] = (real*iters_typeII[scale] + iters) / (real + 1);
        }
    }
    // Write to csv
    std::ofstream file("example.csv");
    file << "gamma,runtime_typeI,iters_typeI,runtime_fast,iters_fast\n";
    for (size_t i = 0; i < gamma.size(); ++i) {
        file << gamma[i] << "," << runtime_typeI[i] << ","  
             << iters_typeI[i] << "," << runtime_typeII[i] << ","
             << iters_typeII[i] << "\n";
    }
    file.close();
    
    // Plot
    Plot2D plot;
    plot.size(600, 400);
    plot.xlabel("\\gamma");
    plot.ylabel("runtime [s]");
    
    // PLot gamma versus runtime
    plot.drawCurve(gamma, runtime_typeI)
        .label("Type I")
        .dashType(1)
        .lineColor("black");
        
    plot.drawCurve(gamma, runtime_typeII)
        .label("Type II")
        .dashType(0)
        .lineColor("red");
    
    // Print the figure
    Figure fig = {{plot}};
    Canvas canvas = {{fig}};
    canvas.size(600, 400);
    //canvas.show();
    canvas.save("example.pdf");

    // Unvectorize
    //CImg<float> X_hat(cols, rows, 1);
    //for (int n = 0; n < N; n++) {
    //    X_hat.data()[n] = x_hat->data[n];
    //}
    //CImgDisplay disp(X_hat, "Estimate");
    //disp.resize(250, 250, true);
    //while (!disp.is_closed()) {
    //    disp.wait();
    //}
    
    gsl_vector_free(x_hat);
    gsl_vector_free(y);
    gsl_vector_free(eta);
    gsl_vector_free(x);
    gsl_matrix_free(H);
    gsl_matrix_free(Lam);
    
    return 0;
}

