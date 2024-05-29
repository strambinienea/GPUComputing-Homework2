#include <cstdlib>
#include <math.h>
#include <iostream>

#define MATRIX_TYPE int
#define MAX_RANDOM_VALUE 100

using namespace std;

/**
 *
 * @param matrix
 * @param transposedMatrix
 * @param matrixSize
 * @return
 */
__global__ void naiveTranspose(const MATRIX_TYPE* matrix, MATRIX_TYPE* transposedMatrix, int matrixSize) {

    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int index = matrixSize * row + column;
    int transposedIndex = matrixSize * column + row;

    if (index < matrixSize) {
        transposedMatrix[transposedIndex] = matrix[index];
    }

}

int main(int argc, char** argv) {

    // Check the size of the matrix is passed as an argument
    if (argc < 4) {
        cout
        << "You must specify the size of the matrix and grid dimension as arguments"
        << endl;

        return -1;
    }

    bool debug = false;
#ifdef DEBUG
    debug = true;
#endif

    // Calculate the size of the matrix and initialize it
    int MATRIX_SIZE = 1 << atoi(argv[1]);

    // The matrix will be divided in GRID_SIZE X GRID_SIZE tiles
    int GRID_SIZE = atoi(argv[2]);

    // Dimension of the tile matrix, each tile is a block
    dim2 GRID_DIMENSION(GRID_SIZE, GRID_SIZE);

    // Dimension of a single tile or block
    dim2 BLOCK_DIMENSION(MATRIX_SIZE / GRID_SIZE, MATRIX_SIZE / GRID_SIZE);


    MATRIX_TYPE *naiveA, *naiveB;

    cudaMallocManaged(&naiveA, MATRIX_SIZE * MATRIX_SIZE * sizeof(MATRIX_TYPE));
    cudaMallocManaged(&naiveB, MATRIX_SIZE * MATRIX_SIZE * sizeof(MATRIX_TYPE));

    initMatrix(naiveA, MATRIX_SIZE);

    naiveTranspose<<GRID_DIMENSION, BLOCK_DIMENSION>>(naiveA, naiveB, MATRIX_SIZE);
    return 0;
}

/**
 * Initialize the matrix with random values
 * @param matrix - The matrix to initialize
 * @param matrixSize - The size of the matrix
 */
void initMatrix(*MATRIX_TYPE matrix, int matrixSize) {

    for (int i = 0; i < matrixSize * matrixSize; i++) {
        matrix[i] = (MATRIX_TYPE) rand() % MAX_RANDOM_VALUE;
    }
}