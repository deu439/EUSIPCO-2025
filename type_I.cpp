#include <iostream>
#include <bits/stdc++.h>
#include <time.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>

#include "anscombe.h"
#include "utils.h"

int type_I_debug(gsl_vector *x_hat, gsl_vector *x, const gsl_vector *y, const gsl_matrix *H, 
           const gsl_matrix *Lam, const gsl_vector *eta, float sigma2, int n_iter,
           std::vector<double> &psnr, std::vector<double> &obj, double &time_per_it)
{
    size_t N = H->size1;    // Size of y
    size_t M = H->size2;    // Size of x
    
    // Calculate the posterior covariance J_inv
    gsl_matrix *EH = gsl_matrix_alloc(N, M);
    gsl_matrix_memcpy(EH, H);
    gsl_matrix_scale_columns(EH, eta);  // EH = EH
    gsl_matrix *J_inv = gsl_matrix_alloc(M, M);
    gsl_matrix_memcpy(J_inv, Lam);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, H, EH, 1.0, J_inv);   // J = H'EH + Lam
    gsl_linalg_cholesky_decomp1(J_inv);
    gsl_linalg_cholesky_invert(J_inv);
    
    // Calculate the x estimator matrix
    gsl_matrix *B = gsl_matrix_alloc(M, N);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, J_inv, EH, 0.0, B);
    gsl_matrix_free(J_inv);

    // Concave term - matrix A = E H J_inv H' E
    gsl_matrix *A = gsl_matrix_alloc(N, N);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, EH, B, 0.0, A);
    gsl_matrix_free(EH);

    // Calculate C
    gsl_matrix *C = gsl_matrix_alloc(N, N);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, H, B, 0.0, C);
    
    // Initialize gamma from observations and record the initial PSNR
    gsl_vector *gamma_hat = gsl_vector_alloc(N);
    gsl_vector *tmp = gsl_vector_alloc(N);
    //gsl_rng *generator = gsl_rng_alloc(gsl_rng_mt19937);
    //gsl_rng_set(generator, 43);
    //for (int n = 0; n<N; n++) {
    //    gamma_hat->data[n] = gsl_rng_uniform(generator) + 1.0;
    //}
    gsl_blas_dgemv(CblasNoTrans, 1.0, H, y, 0.0, tmp);
    for (int n = 0; n < N; n++) {
        gamma_hat->data[n] = f_prime(tmp->data[n], y->data[n], sigma2, eta->data[n]);
    }
    gsl_blas_dgemv(CblasNoTrans, 1.0, B, gamma_hat, 0.0, x_hat);
    
    // Record initial objective value
    psnr.push_back(compute_psnr(x, x_hat));
    double obj_val;
    if (objective(gamma_hat, y, A, eta, sigma2, obj_val) == 1) {
        obj_val = DBL_MAX;
    }
    obj.push_back(obj_val);

    // Start iterating
    time_per_it = 0;
    for (int l = 0; l < n_iter; l++) {
        clock_t start = clock();
        // Estimate gamma 
        gsl_blas_dgemv(CblasNoTrans, 1.0, C, gamma_hat, 0.0, tmp);
        for (int n = 0; n < N; n++) {
            gamma_hat->data[n] = f_prime(tmp->data[n], y->data[n], sigma2, eta->data[n]);
        }
        time_per_it += ((double)(clock() - start)) / CLOCKS_PER_SEC;
        
        // Evaluate PSNR
        gsl_blas_dgemv(CblasNoTrans, 1.0, B, gamma_hat, 0.0, x_hat);
        psnr.push_back(compute_psnr(x, x_hat));

        // Evaluate objective
        if (objective(gamma_hat, y, A, eta, sigma2, obj_val) == 1) {
            obj_val = DBL_MAX;
        }
        obj.push_back(obj_val);
        std::cout << "Iteration " << l+1 << std::endl << std::flush;
    }

    // Compute time per iteration
    time_per_it = time_per_it / n_iter;
    
    std::cout << "Cleaning up" << std::endl << std::flush;
    // Clean up
    gsl_matrix_free(A);
    gsl_matrix_free(B);
    gsl_matrix_free(C);
    gsl_vector_free(gamma_hat);
    gsl_vector_free(tmp);

    return 0;
}

int type_I(gsl_vector *x_hat, const gsl_vector *y, const gsl_matrix *H, 
           const gsl_matrix *Lam, const gsl_vector *eta, float sigma2, int n_iter, double eps)
{
    size_t N = H->size1;    // Size of y
    size_t M = H->size2;    // Size of x
    
    // Calculate the posterior covariance J_inv
    gsl_matrix *EH = gsl_matrix_alloc(N, M);
    gsl_matrix_memcpy(EH, H);
    gsl_matrix_scale_columns(EH, eta);  // EH = EH
    gsl_matrix *J_inv = gsl_matrix_alloc(M, M);
    gsl_matrix_memcpy(J_inv, Lam);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, H, EH, 1.0, J_inv);   // J = H'EH + Lam
    gsl_linalg_cholesky_decomp1(J_inv);
    gsl_linalg_cholesky_invert(J_inv);
    
    // Calculate the x estimator matrix
    gsl_matrix *B = gsl_matrix_alloc(M, N);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, J_inv, EH, 0.0, B);
    gsl_matrix_free(J_inv);

    // Concave term - matrix A = E H J_inv H' E
    gsl_matrix *A = gsl_matrix_alloc(N, N);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, EH, B, 0.0, A);
    gsl_matrix_free(EH);

    // Calculate C
    gsl_matrix *C = gsl_matrix_alloc(N, N);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, H, B, 0.0, C);
    
    // Initialize gamma from observations and record the initial PSNR
    gsl_vector *gamma_hat = gsl_vector_alloc(N);
    gsl_vector *gamma_hat_new = gsl_vector_alloc(N);
    gsl_vector *tmp = gsl_vector_alloc(N);
    gsl_blas_dgemv(CblasNoTrans, 1.0, H, y, 0.0, tmp);
    for (int n = 0; n < N; n++) {
        gamma_hat_new->data[n] = f_prime(tmp->data[n], y->data[n], sigma2, eta->data[n]);
    }
    
    // Start iterating
    double den, num;
    int l;
    for (l = 0; l < n_iter; l++) {
        // Update gamma
        gsl_vector_memcpy(gamma_hat, gamma_hat_new);
        
        // Estimate gamma 
        gsl_blas_dgemv(CblasNoTrans, 1.0, C, gamma_hat, 0.0, tmp);
        for (int n = 0; n < N; n++) {
            gamma_hat_new->data[n] = f_prime(tmp->data[n], y->data[n], sigma2, eta->data[n]);
        }
        
        // Calculate convergence metrics
        den = gsl_blas_dnrm2(gamma_hat);
        gsl_vector_sub(gamma_hat, gamma_hat_new);
        num = gsl_blas_dnrm2(gamma_hat);
        
        // Check for convergence
        if ((num / den) < eps) {
            break; 
        }
        
    }
    
    // Estimate x
    gsl_blas_dgemv(CblasNoTrans, 1.0, B, gamma_hat_new, 0.0, x_hat);

    // Clean up
    gsl_matrix_free(A);
    gsl_matrix_free(B);
    gsl_matrix_free(C);
    gsl_vector_free(gamma_hat);
    gsl_vector_free(tmp);

    return l;
}
