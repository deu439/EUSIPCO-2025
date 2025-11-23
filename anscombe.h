#ifndef ANSCOMBE
#define ANSCOMBE

#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_poly.h>

double inline beta(double z, double y, double sigma2) {
    double s = sigma2 + 3.0/8;
    double tmp = sqrt(y + s) - sqrt(z + s);
    return 2 * tmp * tmp;
}

double inline beta_prime(double z, double y, double sigma2) {
    double s = sigma2 + 3.0/8;
    return 2 - 2*sqrt(y + s) / sqrt(z + s);
}

double inline beta_prime2(double z, double y, double sigma2) {
    double s = sigma2 + 3.0/8;
    double tmp = z + s;
    return sqrt(y + s) / (tmp * sqrt(tmp));
}

double inline f_prime(double z, double y, double sigma2, double eta) {
    double val;
    if (z >= 0) {
        val = z - beta_prime(z, y, sigma2) / eta;
    } else {
        val = z - (beta_prime(0, y, sigma2) + beta_prime2(0, y, sigma2) * z) / eta;
    }
    return val;
}

double inline f_star_prime(double gamma, double y, double sigma2, double eta, double &z) {
    double s = sigma2 + 3.0/8;
    double tmp = gamma + 2/eta;
    double poly[3] = {s - 2*tmp, tmp*tmp - 2*s*tmp, s*tmp*tmp - 4*(y + s)/(eta*eta)};
    double roots[3];
    double *root = NULL;
    int nroots = gsl_poly_solve_cubic(poly[0], poly[1], poly[2], &roots[0], &roots[1], &roots[2]);

    // Find a root that satisfies all conditions
    for (int ind = 0; ind < nroots; ind++) {
        if (roots[ind] < 0) continue;

        if (abs(f_prime(roots[ind], y, sigma2, eta) - gamma) < 1e-10) {
            root = &roots[ind];
            break;
        }
    }

    // We are in the domain of Anscombe function
    if (root != NULL) {
        z = *root;
        return 0;

    // We are in the domain of the quadratic extension
    } else {
        double beta2 = beta_prime2(0, y, sigma2);
        if (abs(eta - beta2) < 1e-10) {
            return 1;
        } else {
            z = eta * gamma + beta_prime(0, y, sigma2) / (eta - beta_prime2(0, y, sigma2));
            return 0;
        }
    }
}


int inline f_star(double gamma, double y, double sigma2, double eta, double &val) {
    double z;
    if (f_star_prime(gamma, y,  sigma2, eta, z) == 1) {
        return 1;
    } else if (z < 0) {
        val = z*gamma - z*z/2 + (beta(0, y, sigma2) + beta_prime(0, y, sigma2)*z + beta_prime2(0, y, sigma2)*z*z/2) / eta;
        return 0;
    } else {
        val = z*gamma - z*z/2 + beta(z, y, sigma2) / eta;
        return 0;
    }
}

#endif

