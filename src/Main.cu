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

    if ( index < matrixSize * matrixSize && transposedIndex < matrixSize * matrixSize ) {

        transposedMatrix[transposedIndex] = matrix[index];
    }
}

__global__ void coalescedTiledTranspose(const MATRIX_TYPE* matrix, MATRIX_TYPE* transposedMatrix, int matrixSize, int tileSize) {
	
	extern __shared__ MATRIX_TYPE tile[];

	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int index = matrixSize * row + column;
    	int transposedIndex = matrixSize * column + row;

	if ( index < matrixSize * matrixSize ) {
		
		tile[column + row * tileSize] = matrix[index];	
    	}
	
	__syncthreads();

	if ( transposedIndex < matrixSize * matrixSize) {
		transposedMatrix[transposedIndex] = tile[column + tileSize * row];
	}
}

/**
 * Initialize the matrix with random values
 * @param matrix - The matrix to initialize
 * @param matrixSize - The size of the matrix
 */
void initMatrix(MATRIX_TYPE* matrix, int matrixSize) {

    for (int i = 0; i < matrixSize * matrixSize; i++) {
        matrix[i] = (MATRIX_TYPE) rand() % MAX_RANDOM_VALUE;
    }
}

void printMatrix(const MATRIX_TYPE* matrix, int matrixSize) {
    
	cout << "Matrix of " << matrixSize << " X " << matrixSize << " elements" << endl;
	for (int i = 0; i < matrixSize * matrixSize; i++) {
		
		if (i % matrixSize == 0) { cout << endl;} 
		cout << matrix[i] << " ";
	}

	cout << endl;
}

int main(int argc, char** argv) {

    // Check the size of the matrix is passed as an argument
    if (argc < 3) {
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
    dim3 GRID_DIMENSION(GRID_SIZE, GRID_SIZE);

    // Dimension of a single tile or block
    const int TILE_SIZE = MATRIX_SIZE / GRID_SIZE;
    dim3 BLOCK_DIMENSION(TILE_SIZE, TILE_SIZE);


    MATRIX_TYPE *naiveA, *naiveB;

    cudaMallocManaged(&naiveA, MATRIX_SIZE * MATRIX_SIZE * sizeof(MATRIX_TYPE));
    cudaMallocManaged(&naiveB, MATRIX_SIZE * MATRIX_SIZE * sizeof(MATRIX_TYPE));

    initMatrix(naiveA, MATRIX_SIZE);

    if (debug) { printMatrix(naiveA, MATRIX_SIZE); }

    cout << "Computing naive matrix transposition of a matrix of size "  
	    << MATRIX_SIZE << " X " << MATRIX_SIZE << " and a grid of size " 
	    << GRID_SIZE << " X " << GRID_SIZE << endl;
    coalescedTiledTranspose<<<GRID_DIMENSION, BLOCK_DIMENSION, TILE_SIZE * TILE_SIZE * sizeof(MATRIX_TYPE)>>>(naiveA, naiveB, MATRIX_SIZE, TILE_SIZE);

    cudaDeviceSynchronize();
    if (debug) { printMatrix(naiveB, MATRIX_SIZE); }
	
    cudaFree(naiveA);
    cudaFree(naiveB);

    return 0;
}

