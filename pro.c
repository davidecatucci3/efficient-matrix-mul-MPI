#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <unistd.h>
#include <string.h>

// I multiply a AxN matrix by a NxB matrix, A, N, and B can be equal or A, B can be equal
// 
// how does it work: exe. mat_mul(4x3, 3x4)=(4,4):
// a11 a12 a13      b11 b12 b13 b14
// a21 a22 a23      b21 b22 b23 b24
// a31 a32 a33      b31 b32 b33 b34
// a41 a42 a43     
// 
// c11 = (a11 * b11) + (a12 * b21) + (a31 * b31)
// c12 = (a11 * b12) + (a12 * b22) + (a31 * 32)
// ...
// c44 = (a41 * b14) + (a42 * b24) + (a43 * b34)
//
// how does it work: exe. mat_mul(4x3, 3x5)=(4,5):
// a11 a12 a13      b11 b12 b13 b14 b15
// a21 a22 a23      b21 b22 b23 b24 b25
// a31 a32 a33      b31 b32 b33 b34 b35
// a41 a42 a43     
// 
// c11 = (a11 * b11) + (a12 * b21) + (a31 * b31)
// c12 = (a11 * b12) + (a12 * b22) + (a31 * 32)
// ...
// c45 = (a41 * b14) + (a42 * b24) + (a43 * b34)
//
// 1. so i need p=A*B independently processes where each process compute N multiplication and (N - 1) additions
// 2. to do it faster i should have p=(A*B)*N independently processes where each processe compute the (A*B)*N multiplications and after => 
// => i have to merge the results using additions
//
// i have 8 cores on my M2 CPU so the AxB must be <= 8 for 1. method and (A*B)*N <= 8 for the 2. method
//
// if AxB < 8 then the missing cores so p_not_used=P_MAX - P_USED can be used to do p_not_used multiplications in the 1. method and for the 2. method they are just lost

// print matrix
void printMatrix(const char* name, int **C, int ROWS, int COLS) {
    printf("%s =\n", name);

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%5d ", C[i][j]); // 5 spaces for alignment
        }
        printf("\n");
    }

    printf("\n");
}

// function to allocate a 2D array dynamically
int **alloc_matrix(int M, int N) {
    int **mat = malloc(M * sizeof(int *));

    for (int i = 0; i < M; i++)
        mat[i] = malloc(N * sizeof(int));

    return mat;
}

// efficient matrix mul 
int **eff_mat_mul(int **A1, int **A2, int A, int B, int N, int r, int n_proc, int mode) {
    if (mode == 0) { // A*B = n_cores
        // a11 a12 a13      b11 b12     c11 (r=0) c12 (r=1)     c00 c01
        // a21 a22 a23      b21 b22     c21 (r=2) c22 (r=3)     c10 c11
        // a31 a32 a33      b31 b32     c31 (r=4) c32 (r=5)     c20 c21
        // a41 a42 a43                  c41 (r=6) c42 (r=7)     c30 c31
        // 
        // c11 = (a11 * b11) + (a12 * b21) + (a12 * b31) => (A1[0][0] * A2[0][0]) + (A1[0][1] * A2[1][0]) + (A1[0][2] * A2[2][0])
        // c12 = (a11 * b12) + (a12 * b22) + (a12 * b32) => (A1[0][0] * A2[0][1]) + (A1[0][1] * A2[1][1]) + (A1[0][2] * A2[2][1])
        // ...
        // c42 = (a41 * b12) + (a42 * b22) + (a42 * b32) => (A1[0][0] * A2[0][0]) + (A1[0][1] * A2[1][0]) + (A1[0][2] * A2[2][0])
        //
        // r=0 -> (A1[0][0] * A2[0][0]) + (A1[0][1] * A2[1][0]) + (A1[0][2] * A2[2][0])
        // r=1 -> (A1[0][0] * A2[0][1]) + (A1[0][1] * A2[1][1]) + (A1[0][2] * A2[2][1])
        // r=2 -> (A1[1][0] * A2[0][0]) + (A1[1][1] * A2[1][0]) + (A1[1][2] * A2[2][0]
        //
        // A1[i][j] -> 0 <= j <= N - 1 
        // A1[i][j] -> 0 <= i <= A - 1 
        // A2[i][j] -> 0 <= j <= B - 1 
        // A2[i][j] -> 0 <= i <= N - 1
        //
        // given r -> (A1[r-1][0] * A2[0][r % N]) + (A1[r-1][1] * A2[1][r % N]) + (A1[r-1][2] * A2[2][r % N])

        int **C = alloc_matrix(A, N);
        int cij = 0;

        if (r == 0) {
            for (int k = 0; k < N; k++) {
                C[0][0] += (A1[0][k] * A2[k][0]);
            }

            for (int source = 1; source < n_proc; source++) {
                MPI_Recv(&cij, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int i = source / B;
                int j = source % B;

                C[i][j] = cij;
            }

            return C;
        } else {
            for (int k = 0; k < N; k++) {
                int i = r / B;
                int j = r % B;

                cij += A1[i][k] * A2[k][j];
            }

            MPI_Send(&cij, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        return C;
    } else if (mode == 1) { // A*B > n_cores
        // r=0 to r=3 (4 cores but operations 8)
        //
        // a11 a12 a13      b11 b12     c11 (r=0) c12 (r=1)     c00 c01
        // a21 a22 a23      b21 b22     c21 (r=2) c22 (r=3)     c10 c11
        // a31 a32 a33      b31 b32     c31 (r=0) c32 (r=1)     c20 c21
        // a41 a42 a43                  c41 (r=2) c42 (r=3)     c30 c31 -> 
        // -> 8/4 = 2 so each cores 2 operations
        //
        // r=0 and r=1 (2 cores but operarions 8)
        //
        // a11 a12 a13      b11 b12     c11 (r=0) c12 (r=1)     c00 c01
        // a21 a22 a23      b21 b22     c21 (r=0) c22 (r=1)     c10 c11
        // a31 a32 a33      b31 b32     c31 (r=0) c32 (r=1)     c20 c21
        // a41 a42 a43                  c41 (r=0) c42 (r=1)     c30 c31 -> 
        // -> 8/2 = 4 so each cores 4 operations
        //
        // r=0 to r=2 (3 cores but operarions 12)
        //
        // a11 a12 a13      b11 b12     c11 (r=0) c12 (r=1) c13 (r=2)    c00 c01 c02
        // a21 a22 a23      b21 b22     c21 (r=0) c22 (r=1) c23 (r=2)    c10 c11 c12
        // a31 a32 a33      b31 b32     c31 (r=0) c32 (r=1) c33 (r=2)    c20 c21 c22
        // a41 a42 a43                  c41 (r=0) c42 (r=1) c43 (r=2)    c30 c31 c32 ->
        // -> 12 / 3 = 4 so each cores 4 operations
        // 
        // r=0 to r=6 (7 cores but 12 operations)
        // a11 a12 a13      b11 b12     c11 (r=0) c12 (r=1) c13 (r=2)    c00 c01 c02
        // a21 a22 a23      b21 b22     c21 (r=3) c22 (r=4) c23 (r=5)    c10 c11 c12
        // a31 a32 a33      b31 b32     c31 (r=6) c32 (r=0) c33 (r=1)    c20 c21 c22
        // a41 a42 a43                  c41 (r=2) c42 (r=3) c43 (r=4)    c30 c31 c32 ->
        // -> 12 / 7 = 1.7 so each core 1.7 operations? -> floor(1.7)=1
        // 
        // they aggregate to which core?
        // given n_proc, r calculate ij -> 4,1=01,21   4,2=10,30    4,3=11,31    4,0=00,20
        // i = r / B; so 1 / 2 = 0 -> i=0, 0+2(4-2)=2
        // j = r % B; so 1 % 2 = 1-> j=1, 

        int **C = alloc_matrix(A, N);
        int cij = 0;
        
        if (r == 0) {

        } else {
            for (int k = 0; k < N; k++) {
                int i = r / B;
                int j = r % B;

                cij += A1[i][k] * A2[k][j];
            }

            MPI_Send(&cij, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

    } else { // A*B < n_cores

    }
}

int main(int argc, char *argv[]) {
    // variables
    int A, B, N;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-A") == 0 && i + 1 < argc) {
            A = atoi(argv[i + 1]); 
        } else if (strcmp(argv[i], "-B") == 0 && i + 1 < argc) {
            B = atoi(argv[i + 1]);
        } else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) {
            N = atoi(argv[i + 1]);
        }
    }

    int **A1 = alloc_matrix(A, N);
    int **A2 = alloc_matrix(N, B);

    // seed the random number generator
    srand(time(NULL));

    // fill matrices with numbers
    for (int i=0; i < A; i++) {
        for (int j = 0; j < N; j++) {
            A1[i][j] = (rand() % 21) - 10;
        }
    }

    for (int i=0; i < B; i++) {
        for (int j = 0; j < N; j++) {
            A2[i][j] = (rand() % 21) - 10;
        }
    }

    int n_cores = sysconf(_SC_NPROCESSORS_ONLN);  // number of processors
    int rank;
    int n_proc;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // perform computation
    if (A * B == n_cores) {        
        int **C1 = eff_mat_mul(A1, A2, A, B, N, rank, n_proc, 0);

        if (rank == 0) {
            printMatrix("Matrix A1", A1, A, N);
            printMatrix("Matrix A2", A2, N, B);
            printMatrix("Result C1", C1, A, B);
        }
    else if (A * B > n_cores) {
        int **C1 = eff_mat_mul(A1, A2, A, B, N, rank, n_proc, 1);

        if (rank == 0) {
            printMatrix("Matrix A1", A1, A, N);
            printMatrix("Matrix A2", A2, N, B);
            printMatrix("Result C1", C1, A, B);
        }
    } else { // A * B < n_cores
        int **C1 = eff_mat_mul(A1, A2, A, B, N, rank, n_proc, 2);

        if (rank == 0) {
            printMatrix("Matrix A1", A1, A, N);
            printMatrix("Matrix A2", A2, N, B);
            printMatrix("Result C1", C1, A, B);
        }
    }

    MPI_Finalize();

    return 0;
}