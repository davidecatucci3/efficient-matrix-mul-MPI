#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <unistd.h>
#include <string.h>

// print matrix
void printMatrix(const char *name, int **C, int ROWS, int COLS) {
    printf("%s =\n", name);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%5d ", C[i][j]); // 5 spaces for alignment
        }
        printf("\n");
    }
    printf("\n");
}

// allocate a 2D array dynamically
int **alloc_matrix(int M, int N) {
    int **mat = malloc(M * sizeof(int *));
    for (int i = 0; i < M; i++)
        mat[i] = malloc(N * sizeof(int));
    return mat;
}

// efficient matrix multiplication
int **eff_mat_mul(int **A1, int **A2, int A, int B, int N, int r, int n_proc, int mode) {
    if (mode == 0) { // A*B = n_cores
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
        int **C = alloc_matrix(A, N);
        int cij;
        int total = A * B;
        int skip = (total + n_proc - 1) / n_proc; // work per process

        if (r == 0) {
            // root process
            for (int x = 0; x < skip; x++) {
                int linear = r * skip + x;
                if (linear >= total) break;

                int i = linear / B;
                int j = linear % B;

                cij = 0;
                for (int k = 0; k < N; k++)
                    cij += A1[i][k] * A2[k][j];

                C[i][j] = cij;
            }

            for (int source = 1; source < n_proc; source++) {
                for (int x = 0; x < skip; x++) {
                    int linear = source * skip + x;
                    if (linear >= total) break;

                    int i = linear / B;
                    int j = linear % B;

                    MPI_Recv(&C[i][j], 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            return C;
        } else {
            // worker process
            for (int x = 0; x < skip; x++) {
                int linear = r * skip + x;
                if (linear >= total) break;

                int i = linear / B;
                int j = linear % B;

                cij = 0;
                for (int k = 0; k < N; k++)
                    cij += A1[i][k] * A2[k][j];

                MPI_Send(&cij, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }

        return C;

    } else { // mode 2: A*B < n_cores
        int **C = alloc_matrix(A, N);
        int cij = 0;
        int n_proc_used = A * B;

        if (r == 0) {
            for (int k = 0; k < N; k++) {
                C[0][0] += (A1[0][k] * A2[k][0]);
            }

            for (int source = 1; source < n_proc_used; source++) {
                MPI_Recv(&cij, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                int i = source / B;
                int j = source % B;
                C[i][j] = cij;
            }

            return C;
        } else if (r < n_proc_used) {
            for (int k = 0; k < N; k++) {
                int i = r / B;
                int j = r % B;
                cij += A1[i][k] * A2[k][j];
            }
            MPI_Send(&cij, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        return C;
    }
}

int main(int argc, char *argv[]) {
    int A = 0, B = 0, N = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-A") == 0 && i + 1 < argc)
            A = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "-B") == 0 && i + 1 < argc)
            B = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc)
            N = atoi(argv[i + 1]);
    }

    int **A1 = alloc_matrix(A, N);
    int **A2 = alloc_matrix(N, B);

    srand(time(NULL));

    for (int i = 0; i < A; i++)
        for (int j = 0; j < N; j++)
            A1[i][j] = (rand() % 21) - 10;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < B; j++)
            A2[i][j] = (rand() % 21) - 10;

    int n_cores = sysconf(_SC_NPROCESSORS_ONLN);
    int rank, n_proc;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &n_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int **C1 = NULL;

    if (A * B == n_cores) {
        C1 = eff_mat_mul(A1, A2, A, B, N, rank, n_proc, 0);
    } else if (A * B > n_cores) {
        C1 = eff_mat_mul(A1, A2, A, B, N, rank, n_proc, 1);
    } else {
        C1 = eff_mat_mul(A1, A2, A, B, N, rank, n_proc, 2);
    }

    if (rank == 0) {
        printMatrix("Matrix A1", A1, A, N);
        printMatrix("Matrix A2", A2, N, B);
        printMatrix("Result C1", C1, A, B);
    }

    MPI_Finalize();
    return 0;
}