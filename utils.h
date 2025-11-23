#ifndef UTILS
#define UTILS

#include <vector>
#include <gsl/gsl_blas.h>
#include <fftw3.h>

void pretty_print(FILE *f,const gsl_matrix *a);
void pretty_print(FILE *f,const gsl_vector *a);
int eye(double alpha, gsl_matrix *out, int k = 0);
void kronecker(const gsl_matrix *K, const gsl_matrix *V, gsl_matrix *H);
double compute_psnr(const gsl_vector *x, const gsl_vector *x_hat);
int objective(const gsl_vector *gamma, const gsl_vector *y, const gsl_matrix *A, const gsl_vector *eta, double sigma2, double &val);
void normalize(std::vector<double>& values);

class DctOrtho2d {
    bool _inv;
    size_t _rows, _cols;
    double *arr1, *arr2;
    fftw_plan plan_y, plan_x;
    
    int orthogonalize_dct(double *arr, size_t rows, size_t cols);
    int orthogonalize_idct(double *arr, size_t rows, size_t cols);
    
public:
    DctOrtho2d(size_t rows, size_t cols, bool inv = false);
    ~DctOrtho2d();
    
    int execute(double *input, double *output);
};

#endif

