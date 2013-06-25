#ifndef MPISVD_H
#define MPISVD_H


typedef struct _rc4_t {
    unsigned char* state;
    int i;
    int j;
} rc4_t;

typedef struct _compressed_matrix_t {
    unsigned int **columns;
    size_t rows;
    size_t cols;
    size_t col_offset;
} compressed_matrix_t;

typedef struct _factors_t {
    double *ifactors;
    double *ufactors;
    size_t dims;
    size_t rows;
    size_t cols;
    size_t iterations;
    size_t *progress;
} factors_t;

unsigned int **load_matrix(int rank, int size, int rows, int cols, int& start, int& count);
void swap_byte_order(unsigned int* n);
void swap_byte_order(unsigned int* n, int size);
compressed_matrix_t *load_matrix(int rank, int size, int rows, int cols);
int unpacked_column_size(unsigned int* column);
int packed_matrix_size(unsigned int** matrix, int cols);

factors_t* prepare_factors(char* seed, int dimensions, int rows, int cols);
void free_factors(factors_t *factors);
inline void update_value(factors_t*& factors, size_t& row, size_t& col, double& value, double& gamma, double& err);
void update_factors(compressed_matrix_t *matrix, factors_t *factors, int phase, int size, double gamma, double& err, size_t& total);

rc4_t* init_random(char* seed);
void free_random(rc4_t *random);
int rand_char(rc4_t *random);
double rand_double(rc4_t *random);
double rand_gaussian(rc4_t *random);


#endif // MPISVD_H
