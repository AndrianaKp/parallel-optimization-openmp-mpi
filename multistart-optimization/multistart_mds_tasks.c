#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>

#define MAXVARS     (250)   /* max # of variables        */
#define EPSMIN      (1E-6)  /* ending value of stepsize  */

/* prototype of local optimization routine, code available in torczon.c */
extern void mds(double *startpoint, double *endpoint, int n, double *val, double eps, int maxfevals, int maxiter,
         double mu, double theta, double delta, int *ni, int *nf, double *xl, double *xr, int *term);

/* global variables */
unsigned long funevals = 0;

/* Rosenbrock classic parabolic valley ("banana") function */
double f(double *x, int n)
{
    double fv;
    int i;

    #pragma omp atomic
    funevals++;
    
    fv = 0.0;
    for (i=0; i<n-1; i++)   /* rosenbrock */
        fv = fv + 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);

        usleep(100);    /* do not remove, introduces some artificial work */

    return fv;
}

double get_wtime(void)
{
    struct timeval t;

    gettimeofday(&t, NULL);

    return (double)t.tv_sec + (double)t.tv_usec*1.0e-6;
}

int main(int argc, char *argv[])
{
    /* problem parameters */
    int nvars = 4;      /* number of variables (problem dimension) */
    int ntrials = 64;   /* number of trials */
    double lower[MAXVARS], upper[MAXVARS];  /* lower and upper bounds */

    /* mds parameters */
    double eps = EPSMIN;
    int maxfevals = 10000;
    int maxiter = 10000;
    double mu = 1.0;
    double theta = 0.25;
    double delta = 0.25;

    /* information about the best point found by multistart */
    double best_pt[MAXVARS];
    double best_fx = 1e10;
    int best_trial = -1;
    int best_nt = -1;
    int best_nf = -1;

    /* initialization of lower and upper bounds of search space */
    for (int i = 0; i < MAXVARS; i++) lower[i] = -2.0;  /* lower bound: -2.0 */
    for (int i = 0; i < MAXVARS; i++) upper[i] = +2.0;  /* upper bound: +2.0 */

    double t0 = get_wtime();

    #pragma omp parallel
    {
     #pragma omp single
     {
            printf("OpenMP threads = %d\n", omp_get_num_threads());
            
            for (int trial = 0; trial < ntrials; trial++) {

                #pragma omp task firstprivate(trial)
             {
                    /* --- thread-safe RNG per trial using erand48 --- */
                    unsigned short xsubi[3];
                    xsubi[0] = (unsigned short)(trial & 0xFFFF);
                    xsubi[1] = (unsigned short)((trial * 1103515245u + 12345u) & 0xFFFF);
                    xsubi[2] = (unsigned short)((trial * 214013u + 2531011u) & 0xFFFF);

                    /* trial-local arrays/vars */
                    double startpt_local[MAXVARS];
                    double endpt_local[MAXVARS];
                    double fx_local;
                    int nt_local, nf_local;
                    int term_local = -1;

                    /* random start point in [-2,2) */
                    for (int i = 0; i < nvars; i++) {
                        startpt_local[i] = lower[i] + (upper[i]-lower[i]) * erand48(xsubi);
                }

                    mds(startpt_local, endpt_local, nvars, &fx_local, eps, maxfevals, maxiter,
                        mu, theta, delta, &nt_local, &nf_local, lower, upper, &term_local);
                  /* update global best safely */
                  #pragma omp critical
                    {
                        if (fx_local < best_fx) {
                            best_fx = fx_local;
                            best_trial = trial;
                            best_nt = nt_local;
                            best_nf = nf_local;
                            for (int i = 0; i < nvars; i++)
                                best_pt[i] = endpt_local[i];
                      }
                 }
                } /* end task */
            } /* end loop */

            #pragma omp taskwait
        } /* end single */
    } /* end parallel */

    double t1 = get_wtime();

    printf("\n\nFINAL RESULTS:\n");
    printf("Elapsed time = %.3lf s\n", t1-t0);
    printf("Total number of trials = %d\n", ntrials);
    printf("Total number of function evaluations = %ld\n", funevals);
    printf("Best result at trial %d used %d iterations, %d function calls and returned\n", best_trial, best_nt, best_nf);
    for (int i = 0; i < nvars; i++) {
        printf("x[%3d] = %15.7le \n", i, best_pt[i]);
    }
    printf("f(x) = %15.7le\n", best_fx);

    return 0;
}