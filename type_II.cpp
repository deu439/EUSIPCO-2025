#include <iostream>
#include <unistd.h>
#include <bits/stdc++.h>
#include <time.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_poly.h>

#include "anscombe.h"
#include "utils.h"

inline double solve_for_gamma(double a, double b, double y, double sigma2, double eta) {
    double s = sigma2 + 3.0 / 8;
    double term1 = eta*(eta-a) / (2*a);
    term1 = term1 * term1;
    double term2 = eta*(eta-a)*(eta*b - 4*a)/(4*a*a);
    double term3 = (eta*b - 4*a) / (4*a);
    term3 = term3 * term3;

    double poly[3] = {(s*term1 - term2) / term1, (term3 - s*term2) / term1,  (s*term3 - y - s) / term1};
    double roots[3];
    double *root = &roots[0];
    if (gsl_poly_solve_cubic(poly[0], poly[1], poly[2], &roots[0], &roots[1], &roots[2]) > 1) {
        root = &roots[2];
        while (*root < 0) root--;
    }
    //double gamma = -1;
    //if (gsl_poly_solve_cubic(poly[0], poly[1], poly[2], &roots[0], &roots[1], &roots[2]) > 1) {
    //    // Find a root that satisfies all conditions
    //    for (int i = 0; i < 3; i++) {
    //        if (roots[i] < 0) continue;
    //
    //        gamma = (2*eta*(roots[i]) - b) / (2*a);
    //        if (abs(f_prime(roots[i], y, sigma2, eta) - gamma) < 1e-10) break;
    //    }
    //}
    
    return (2*eta*(*root) - b) / (2*a);
}

int type_II_debug(gsl_vector *x_hat, gsl_vector *x, const gsl_vector *y, const gsl_matrix *H, 
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
    
    // Calculate the estimator matrix B = J_inv H' E
    gsl_matrix *B = gsl_matrix_alloc(M, N);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, J_inv, EH, 0.0, B);
    gsl_matrix_free(J_inv);
    
    // Concave term - matrix A = E H J_inv H' E
    gsl_matrix *A = gsl_matrix_alloc(N, N);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, EH, B, 0.0, A);
    gsl_matrix_free(EH);
    
    double a, b;
    gsl_vector *gamma_hat = gsl_vector_alloc(N);
    gsl_vector *tmp = gsl_vector_alloc(N);
    // Initialize gamma randomly
    //gsl_rng *generator = gsl_rng_alloc(gsl_rng_mt19937);
    //gsl_rng_set(generator, 43);
    //for (int n = 0; n<N; n++) {
    //    gamma_hat->data[n] = gsl_rng_uniform(generator) + 1.0;
    //}

    // Initialize gamma from observations and record initial PSNR value
    gsl_blas_dgemv(CblasNoTrans, 1.0, H, y, 0.0, tmp);
    for (int n = 0; n < N; n++) {
        gamma_hat->data[n] = f_prime(tmp->data[n], y->data[n], sigma2, eta->data[n]);
    }
    gsl_blas_dgemv(CblasNoTrans, 1.0, B, gamma_hat, 0.0, x_hat);
    psnr.push_back(compute_psnr(x, x_hat));

    // Record initial objective value
    double obj_val;
    if (objective(gamma_hat, y, A, eta, sigma2, obj_val) == 1) {
        obj_val = DBL_MAX;
    }
    obj.push_back(obj_val);

    // Start iterating
    time_per_it = 0;
    for (int l = 0; l < n_iter; l++) {
        // Estimate gamma 
        clock_t start = clock();
        for (int n = 0; n < N; n++) {
            a = gsl_matrix_get(A, n, n);
            gsl_vector_const_view v = gsl_matrix_const_column(A, n);
            gsl_blas_ddot(&v.vector, gamma_hat, &b);
            gamma_hat->data[n] = solve_for_gamma(a, 2*(b - a*gamma_hat->data[n]), y->data[n], sigma2, eta->data[n]);
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
    
    // Clean up
    gsl_matrix_free(B);
    gsl_matrix_free(A);
    gsl_vector_free(gamma_hat);
    gsl_vector_free(tmp);
    
    return 0;
}


int type_II(gsl_vector *x_hat, const gsl_vector *y, const gsl_matrix *H, 
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
    
    // Calculate the estimator matrix B = J_inv H' E
    gsl_matrix *B = gsl_matrix_alloc(M, N);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, J_inv, EH, 0.0, B);
    gsl_matrix_free(J_inv);
    
    // Concave term - matrix A = E H J_inv H' E
    gsl_matrix *A = gsl_matrix_alloc(N, N);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, EH, B, 0.0, A);
    gsl_matrix_free(EH);
    
    double a, b;
    gsl_vector *gamma_hat = gsl_vector_alloc(N);
    gsl_vector *gamma_hat_new = gsl_vector_alloc(N);
    gsl_vector *tmp = gsl_vector_alloc(N);

    // Initialize gamma from observations and record initial PSNR value
    gsl_blas_dgemv(CblasNoTrans, 1.0, H, y, 0.0, tmp);
    for (int n = 0; n < N; n++) {
        gamma_hat_new->data[n] = f_prime(tmp->data[n], y->data[n], sigma2, eta->data[n]);
    }

    // Start iterating
    double num, den;
    int l;
    for (l = 0; l < n_iter; l++) {
        // Update gamma
        gsl_vector_memcpy(gamma_hat, gamma_hat_new);
        
        // Estimate gamma 
        for (int n = 0; n < N; n++) {
            a = gsl_matrix_get(A, n, n);
            gsl_vector_const_view v = gsl_matrix_const_column(A, n);
            gsl_blas_ddot(&v.vector, gamma_hat_new, &b);
            gamma_hat_new->data[n] = solve_for_gamma(a, 2*(b - a*gamma_hat_new->data[n]), y->data[n], sigma2, eta->data[n]);
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
    gsl_matrix_free(B);
    gsl_matrix_free(A);
    gsl_vector_free(gamma_hat);
    gsl_vector_free(tmp);
    
    return l;
}
