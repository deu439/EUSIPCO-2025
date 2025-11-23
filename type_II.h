#ifndef TYPE_II
#define TYPE_II

#include <gsl/gsl_blas.h>
#include <bits/stdc++.h>

int type_II_debug(gsl_vector *x_hat, gsl_vector *x, const gsl_vector *y, const gsl_matrix *H, 
            const gsl_matrix *Lam, const gsl_vector *eta, float sigma2, int n_iter,
            std::vector<double> &psnr, std::vector<double> &obj, double &time_per_it);

int type_II(gsl_vector *x_hat, const gsl_vector *y, const gsl_matrix *H, 
            const gsl_matrix *Lam, const gsl_vector *eta, float sigma2, int n_iter, double eps);

#endif
