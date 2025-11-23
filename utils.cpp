#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>
#include <gsl/gsl_blas.h>
#include <math.h>
#include <float.h>
#include <fftw3.h>

#include "anscombe.h"
#include "utils.h"

void pretty_print(FILE *f,const gsl_matrix *a)
{
    int i, j;
    fprintf(f,"\n\t");
    for (j = 0; j < a->size2; j++)
    {
        fprintf(f,"(%d)\t",j);
    }
    fprintf(f,"\n");
    for (i = 0; i < a->size1; i++)
    {
        fprintf(f,"(%d)\t", i);
        for (j = 0; j < a->size2; j++)
        {
            fprintf(f,"%g\t", gsl_matrix_get(a, i, j));
        }
        fprintf(f,"\n");
    }
}

void pretty_print(FILE *f,const gsl_vector *a)
{
    int i, j;
    fprintf(f,"\n\t");
    for (j = 0; j < a->size; j++)
    {
        fprintf(f,"(%d)\t",j);
    }
    fprintf(f,"\n");
    for (i = 0; i < a->size; i++)
    {
        fprintf(f, "%g\t", gsl_vector_get(a, i));
    }
    fprintf(f,"\n");
}

int eye(double alpha, gsl_matrix *out, int k) {
/* Calculate out = I + alpha*out where I is the identity matrix. */
    size_t n = out->size1;
    gsl_matrix_scale(out, alpha);
    for (int i = 0; i < n - abs(k); i++) {
        out->data[(k > 0)*k - (k < 0)*n*k + i*n + i] += 1.0;
    }
    return 0;
}

void kronecker(const gsl_matrix *K, const gsl_matrix *V, gsl_matrix *H) 
{
    for (size_t i=0; i<K->size1; i++) {
        for (size_t j=0; j<K->size2; j++) {
            gsl_matrix_view H_sub=gsl_matrix_submatrix(H, i*V->size1, j*V->size2, V->size1, V->size2);
            gsl_matrix_memcpy(&H_sub.matrix, V);
            gsl_matrix_scale(&H_sub.matrix, gsl_matrix_get(K, i, j));
        }
    }
    return;
}

double compute_psnr(const gsl_vector *x, const gsl_vector *x_hat) {
    double mse = 0;
    double max = -DBL_MAX;
    double serr;
    for (int n = 0; n < x->size; n++) {
        serr = x->data[n] - x_hat->data[n];
        mse += serr * serr;
        if (x->data[n] > max) max = x->data[n];
    }
    return 20*log10(max / sqrt(mse / x->size));
}

int objective(const gsl_vector *gamma, const gsl_vector *y, const gsl_matrix *A, const gsl_vector *eta, double sigma2, double &val) {
    int N = gamma->size;
    double term1 = 0;
    for (int n = 0; n < N; n++) {
        if (f_star(gamma->data[n], y->data[n], sigma2, eta->data[n], val) == 1) {
            return 1;
        }
        term1 += 2 * eta->data[n] * val;
    }

    double term2;
    gsl_vector *tmp = gsl_vector_alloc(N);
    gsl_blas_dgemv(CblasNoTrans, 1.0, A, gamma, 0.0, tmp);
    gsl_blas_ddot(tmp, gamma, &term2);
    gsl_vector_free(tmp);

    val = term1 - term2;
    return 0;
}


void normalize(std::vector<double>& values) {
    if (values.empty()) return;

    double minVal = *std::min_element(values.begin(), values.end());
    double maxVal = *std::max_element(values.begin(), values.end());

    if (maxVal == minVal) {
        std::fill(values.begin(), values.end(), 0.0); // Avoid division by zero
        return;
    }

    for (double& val : values) {
        val = (val - minVal) / (maxVal - minVal);
    }
}
