# efficient-matrix-mul-MPI
## 🧩 Overview
This project implements an **efficient matrix multiplication** algorithm using **MPI (Message Passing Interface)**.  
The goal is to perform parallel matrix multiplication while **optimally managing the number of active processes** relative to the number of available CPU cores.

The program automatically detects the number of cores available on the machine (e.g., 8 on an M2 CPU) and adjusts its parallelization strategy to maximize performance without wasting computational resources.

---

## 🚀 Description
This program multiplies two matrices:

- `A1`: an **A×N** matrix  
- `A2`: an **N×B** matrix  

The resulting matrix `C` is of size **A×B**.

There are three computation modes, automatically chosen based on matrix size and available cores:

1. **Mode 0** (`A×B == n_cores`):  
   - Each process computes **one element** of the result matrix independently.  
   - All processes perform exactly one multiplication + addition loop for a single `C[i][j]`.  
   - Results are gathered on the root process.

2. **Mode 1** (`A×B > n_cores`):  
   - When there are **more operations than available cores**, each process handles **multiple matrix elements**.  
   - Workload is evenly distributed using integer division `(A×B)/n_proc`.  
   - Results are merged efficiently through MPI communication.

3. **Mode 2** (`A×B < n_cores`):  
   - When there are **fewer operations than cores**, some cores remain idle.  
   - Future improvements may include assigning extra multiplications to idle cores.

---

## 🔬 Efficiency Concept
Typical MPI matrix multiplication launches one process per matrix cell or per row.  
This implementation introduces an **adaptive strategy**:

- If the number of required operations (`A×B`) is smaller than or equal to the number of cores, each process handles one element.
- If more operations exist than available cores, tasks are **divided dynamically** so that each process computes multiple cells in the result matrix.
- The goal is to **avoid idle processes** and **minimize communication overhead**.

This approach ensures efficient CPU utilization whether matrices are small, large, square, or rectangular.

---

## 🧠 Example

For example, multiplying:

## 🧩 Compilation

Use `mpicc` to compile the program:

```bash
mpicc -o matmul mpi_matmul.c
mpirun -np 8 ./matmul -A 4 -B 4 -N 3
```
