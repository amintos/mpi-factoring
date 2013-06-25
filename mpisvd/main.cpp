
#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <math.h>

#include <mpich2/mpi.h>

#include "main.h"


using namespace std;

#define DEFAULT_COLUMNS 17700
#define DEFAULT_ROWS 500000
#define DEFAULT_DIMS 10

int main(int argc, char *argv[])
{
    int mpi_rank, mpi_size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    cout << "This is instance " << mpi_rank << " of " << mpi_size << endl;

    compressed_matrix_t *input;
    input = load_matrix(mpi_rank, mpi_size, DEFAULT_ROWS, DEFAULT_COLUMNS);
    cout <<  "Offset: " << input->col_offset
         << " Columns: " << input->cols
         << " Elements: " << packed_matrix_size(input->columns, input->cols) << endl;

    cout << "This is instance " << mpi_rank << " of " << mpi_size << endl;
    cout << "Initializing factor model..." << endl;
    factors_t *factors = prepare_factors("seed123", DEFAULT_DIMS, DEFAULT_ROWS, DEFAULT_COLUMNS);

    double error;
    size_t total;

    for (int i = 0; i < 10; ++i) {
        error = 0.0;
        total = 0;

        cout << mpi_rank << ": factorizing Block " << (i + mpi_rank) % mpi_size << endl;

        update_factors(input, factors,
                       (i + mpi_rank) % mpi_size, /* partition index offset by mpi_rank -> diagonal */
                       mpi_size,                  /* number of partitions */
                       0.01 * pow(0.99, i), error, total);

        cout << mpi_rank << ": Local RSME = " << sqrt(error / (double)total) << endl;
    }

    cout << "Ok.\n";

    free_factors(factors);
    return 0;
}


// ----------------------------------------------------------------------------
// Compressed Sparse Column import
// Example Folder:
//      ./matrix/0  ... ./matrix/100  for 100 elements
// File (in Big Endian Byte Order!):
//      Bytes 0 .. 3       = Number of following 4-byte integers
//      Bytes 4n .. 4n + 3 = Integer encoding a "Run".
//          Run: upper 24 bits = Gap of zeroes before entry,
//               lower  8 bits = The entry itself
// ----------------------------------------------------------------------------

int load_column(unsigned int** column, int index) {
    char filename[10];
    unsigned int size;

    sprintf(filename, "./matrix/%i", index);

    ifstream file(filename, ios::in | ios::binary);

    // file format containing 4-byte big endian integers
    // but mpisvd is supposed to run on little endian system

    file.read((char*)&size, 4);
    int rand_char(rc4_t *random);
    swap_byte_order(&size);

    // allocate one element over
    *column = (unsigned int*)malloc((1 + size) * 4);
    file.read((char*)(*column), size * 4);
    swap_byte_order(*column, size);

    // terminate column vector by zero entry
    // which never occurs in compressed column format
    (*column)[size] = 0;

    file.close();
    //cout << "Read column " << index << ".Bytes packed: " << size * 4 << endl; //<< " / unpacked: " << unpacked_column_size(*column) <<  endl;
    return size;
}

int unpacked_column_size(unsigned int* column) {
    int i = 0, n = 0;

    while (column[i] != 0) {
        n += column[i] >> 8;
        ++i;
    }

    return n;
}

int packed_column_size(unsigned int* column) {
    int i = 0;

    while (column[i] != 0) {
        ++i;
    }

    return i;
}

int packed_matrix_size(unsigned int** matrix, int cols) {
    int n = 0;
    for (int i = 0; i < cols; ++i) {
        n += packed_column_size(matrix[i]);
    }
    return n;
}

void swap_byte_order(unsigned int* n) {
    *n =  ((*n & 0xff000000) >> 24)
        | ((*n & 0x00ff0000) >> 8 )
        | ((*n & 0x0000ff00) << 8 )
        | ((*n & 0x000000ff) << 24);
}

void swap_byte_order(unsigned int* n, int size) {
    for (int i = 0; i < size; ++i) {
        swap_byte_order(&n[i]);
    }
}

compressed_matrix_t *load_matrix(int rank, int size, int rows, int cols) {
    unsigned int **columns;

    size_t start = (cols / size) * rank;
    size_t count = (cols / size) + ((rank == size - 1) ? cols % size : 0);

    columns = (unsigned int**)malloc(sizeof(unsigned int*) * count);

    for (size_t i = 0; i < count; ++i) {
        load_column(&columns[i], start + i);
    }

    compressed_matrix_t *result = (compressed_matrix_t*)malloc(sizeof(compressed_matrix_t));

    result->cols = count;
    result->col_offset = start;
    result->columns = columns;
    result->rows = rows;

    return result;
}



// ----------------------------------------------------------------------------
// ARCFOUR PRNG with normal distribution support
// ----------------------------------------------------------------------------
rc4_t* init_random(char* seed) {
    rc4_t* result = (rc4_t*)malloc(sizeof(rc4_t));
    int j = 0;
    int k = strlen(seed);
    unsigned char* s = (unsigned char*)malloc(sizeof(unsigned char) * 256);

    for (int i = 0; i < 256; ++i) {
        s[i] = i;
    }

    for (int i = 0; i < 256; ++i) {
        int temp;

        j = (j + s[i] + seed[i % k]) % 256;
        temp = s[i];
        s[i] = s[j];
        s[j] = temp;
    }

    result->state = s;
    result->i = 0;
    result->j = 0;

    return result;
}

void free_random(rc4_t *random) {
    free(random->state);
    free(random);
}

int rand_char(rc4_t *random) {
    unsigned char temp, result;
    unsigned char *s = random->state;
    int i = random->i = (random->i + 1) % 256;
    int j = random->j = (random->j + s[i]) % 256;

    result = s[(int)(s[i] ^ s[j])];

    temp = s[i];
    s[i] = s[j];
    s[j] = temp;

    return result;
}

// -> Double between incl. 0.0 and excl. 1.0 with 32 bits entropy
double rand_double(rc4_t *random) {
    return (double)rand_char(random) * 0.0039062500000000e-00 +
           (double)rand_char(random) * 1.5258789062500000e-05 +
           (double)rand_char(random) * 5.9604644775390630e-08 +
           (double)rand_char(random) * 2.3283064365386963e-10;
}

// Box-Muller Transform (-> double with 64 bits entropy)
double rand_gaussian(rc4_t *random) {
    return sin(6.283185307179586 * rand_double(random))
            * sqrt(-2.0 * log(rand_double(random)));
}

// ----------------------------------------------------------------------------
// Matrix Factorization
// ----------------------------------------------------------------------------

// Initialize factor matrices with gaussian noise
factors_t* prepare_factors(char* seed, int dimensions, int rows, int cols) {
    size_t isize = (size_t)cols * (size_t)dimensions;
    size_t usize = (size_t)rows * (size_t)dimensions;
    double* umat = (double*)malloc(sizeof(double) * usize);
    double* imat = (double*)malloc(sizeof(double) * isize);

    rc4_t *prng = init_random(seed);

    for (size_t i = 0; i < isize; ++i) {
        imat[i] = rand_gaussian(prng);
    }

    for (size_t i = 0; i < usize; ++i) {
        umat[i] = rand_gaussian(prng);
    }

    factors_t *result = (factors_t*)malloc(sizeof(factors_t));
    result->cols = cols;
    result->rows = rows;
    result->ifactors = imat;
    result->ufactors = umat;
    result->dims = dimensions;
    result->iterations = 0;
    result->progress = (size_t*)malloc(sizeof(size_t) * cols);

    for (int i =0; i < cols; ++i) {
        result->progress[i] = 0;
    }

    free_random(prng);

    return result;
}

void free_factors(factors_t *factors) {
    free(factors->ifactors);
    free(factors->ufactors);
    free(factors->progress);
    free(factors);
}

inline void update_value(factors_t*& factors, size_t& row, size_t& col, double& value, double& gamma, double& err) {
    double *u_row = &(factors->ufactors[row * factors->dims]);
    double *i_col = &(factors->ifactors[col * factors->dims]);

    // prediction
    double dot_product = 0.0;
    for (size_t i = 0; i < factors->dims; ++i) {
        dot_product += u_row[i] * i_col[i];
    }

    // prediction error
    double error = value - dot_product;

    // gradient descend
    for (size_t i = 0; i < factors->dims; ++i) {
        i_col[i] += gamma * u_row[i] * error;
        u_row[i] += gamma * i_col[i] * error;
    }

    // aggregate debug info (for mean squared error)
    err += error * error;
}

// Search for starting positions in each column
// too bad this is O(n)
void start_positions(compressed_matrix_t *matrix, int phase, int size, size_t *indices) {
    size_t start = phase * (matrix->rows / size);
    for (int col = 0; col < matrix->cols; ++col) {
        size_t i = 0;
        size_t row = 0;
        unsigned int* column = matrix->columns[col];

        while (row < start) {
            row += column[i] >> 8;
            i++;
        }

        indices[col] = i;
    }
}

// Update factors for a single block
void update_factors(compressed_matrix_t *matrix, factors_t *factors, int phase, int size, double gamma, double& err, size_t& total) {
    size_t start = (matrix->rows / size) * phase;
    size_t count = (matrix->rows / size)
            + ((phase == size - 1) ? (matrix->rows % size) : 0);

    size_t *prog = (size_t*)malloc(sizeof(size_t) * matrix->cols);
    size_t maxrow = start + count;
    size_t elements = 0;
    double error = 0.0;

    // search RLE column for initial index this block is concerned with
    start_positions(matrix, phase, size, prog);

    // traverse columns
    for (size_t col = 0; col < matrix->cols; ++col) {

        size_t row = 0;                 // decompressed absolute index
        size_t row_index = prog[col];   // RLE-index in compressed column
        unsigned int *column = matrix->columns[col];

        // traverse rows (stay inside block!)
        while (row < maxrow) {
            // extract number of zeroes and following value from entry

            if (column[row_index] == 0) {
                // we're done here
                break;
            }

            size_t zeroes = column[row_index] >> 8;
            double value = (double)(column[row_index] & 0xff);

            update_value(factors, row, col, value, gamma, error);

            row_index++;
            elements++;
            row += zeroes + 1; // + 1: the non-zero entry itself
        }

        prog[col] = row_index;          // store for next block
    }

    // mean squared error
    err += error;
    total += elements;
}

