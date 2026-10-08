#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h>

#define MAXVARS     (250)
#define EPSMIN      (1E-6)

/* prototype of local optimization routine, code available in torczon.c */
extern void mds(double *startpoint, double *endpoint, int n, double *val,
                double eps, int maxfevals, int maxiter, double mu, double theta,
                double delta, int *ni, int *nf, double *xl, double *xr, int *term);

/* Each MPI rank has its own copy */
unsigned long funevals = 0;

/* Rosenbrock function */
double f(double *x, int n)
{
    double fv = 0.0;
    int i;

    funevals++;

    for (i = 0; i < n - 1; i++)
        fv += 100.0 * pow((x[i + 1] - x[i] * x[i]), 2) + pow((x[i] - 1.0), 2);

    usleep(100);
    return fv;
}

double get_wtime(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nvars = 4;
    int ntrials = 64;

    double lower[MAXVARS], upper[MAXVARS];
    for (int i = 0; i < MAXVARS; i++) lower[i] = -2.0;
    for (int i = 0; i < MAXVARS; i++) upper[i] = +2.0;

    /* mds parameters */
    double eps = EPSMIN;
    int maxfevals = 10000;
    int maxiter  = 10000;
    double mu = 1.0, theta = 0.25, delta = 0.25;

    double startpt[MAXVARS], endpt[MAXVARS];
    double fx;
    int nt, nf;

    double best_pt[MAXVARS];
    double best_fx = 1e10;
    int best_trial = -1;
    int best_nt = -1;
    int best_nf = -1;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = get_wtime();

    for (int trial = rank; trial < ntrials; trial += size) {

        unsigned short xsubi[3];
        xsubi[0] = (unsigned short)(trial & 0xFFFF);
        xsubi[1] = (unsigned short)((trial * 1103515245u + 12345u) & 0xFFFF);
        xsubi[2] = (unsigned short)((trial * 214013u + 2531011u) & 0xFFFF);

        for (int i = 0; i < nvars; i++) {
            startpt[i] = lower[i] + (upper[i] - lower[i]) * erand48(xsubi);
        }

        int term = -1;
        mds(startpt, endpt, nvars, &fx, eps, maxfevals, maxiter,
            mu, theta, delta, &nt, &nf, lower, upper, &term);

        if (fx < best_fx) {
            best_fx = fx;
            best_trial = trial;
            best_nt = nt;
            best_nf = nf;
            for (int i = 0; i < nvars; i++)
                best_pt[i] = endpt[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = get_wtime();

    /* Find global minimum using MINLOC */
    struct { double value; int rank; } local_min, global_min;
    local_min.value = best_fx;
    local_min.rank  = rank;

    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    int winner = -1;
    if (rank == 0) winner = global_min.rank;
    MPI_Bcast(&winner, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Winner sends its best to rank 0 if needed */
    if (rank == winner && rank != 0) {
        MPI_Send(best_pt, nvars, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&best_trial, 1, MPI_INT,    0, 1, MPI_COMM_WORLD);
        MPI_Send(&best_nt,    1, MPI_INT,    0, 2, MPI_COMM_WORLD);
        MPI_Send(&best_nf,    1, MPI_INT,    0, 3, MPI_COMM_WORLD);
        MPI_Send(&best_fx,    1, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        if (winner != 0) {
            MPI_Recv(best_pt,   nvars, MPI_DOUBLE, winner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&best_trial, 1,   MPI_INT,    winner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&best_nt,    1,   MPI_INT,    winner, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&best_nf,    1,   MPI_INT,    winner, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&best_fx,    1,   MPI_DOUBLE, winner, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            best_fx = global_min.value;
        }
    }

    /* Sum total function evaluations across ranks */
    unsigned long total_funevals = 0;
    MPI_Reduce(&funevals, &total_funevals, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n\nFINAL RESULTS:\n");
        printf("Elapsed time = %.3lf s\n", t1 - t0);
        printf("Total number of trials = %d\n", ntrials);
        printf("Total number of function evaluations = %lu\n", total_funevals);
        printf("Best result at trial %d used %d iterations, %d function calls and returned\n",
               best_trial, best_nt, best_nf);

        for (int i = 0; i < nvars; i++)
            printf("x[%3d] = %15.7le \n", i, best_pt[i]);

        printf("f(x) = %15.7le\n", best_fx);
    }
    MPI_Finalize();
    return 0;
}