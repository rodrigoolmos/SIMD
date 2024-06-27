#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h> // AVX2
#include <time.h>

#define ROWS 512 // Definir número de filas de la matriz
#define COLS 64 // Definir número de columnas de la matriz

static float matrix[ROWS * COLS];
static float transposeSMD[COLS * ROWS];
static float transpose[COLS * ROWS];

void fill_random_matrix(float* matrix, int rows, int cols);
void transpose_matrix_SMD(float* matrix, float* result, int rows, int cols);
void transpose_matrix(float* matrix, float* result, int rows, int cols);
void print_matrix(const char* name, float* matrix, int rows, int cols);
void check_integrity(float* result1, float* result2, int rows, int cols);

int main() {
    srand(time(NULL));

    // Llenar matriz con valores aleatorios
    fill_random_matrix(matrix, ROWS, COLS);
    
    while(1){
        // Medir tiempo de ejecución y realizar la transposición SIMD
        clock_t start = clock();
        transpose_matrix_SMD(matrix, transposeSMD, ROWS, COLS);
        clock_t end = clock();
        double time_smd = (double)(end - start) / CLOCKS_PER_SEC;

        // Medir tiempo de ejecución y realizar la transposición convencional
        start = clock();
        transpose_matrix(matrix, transpose, ROWS, COLS);
        end = clock();
        double time_standard = (double)(end - start) / CLOCKS_PER_SEC;

        // Verificar la integridad de los datos
        check_integrity(transposeSMD, transpose, ROWS, COLS);

        // Imprimir tiempos de ejecución
        printf("Tiempo de ejecución (SMD): %f segundos\n", time_smd);
        printf("Tiempo de ejecución (Standard): %f segundos\n", time_standard);
    }

    return 0;
}

void fill_random_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i*cols + j] = i*cols + j; 
        }
    }
}

void transpose_matrix_SMD(float* matrix, float* result, int rows, int cols) {
    for (int i = 0; i < rows; i += 8) {
        for (int j = 0; j < cols; j++) {
            __m256 vec = _mm256_setr_ps(
                matrix[i * cols + j],
                matrix[(i + 1) * cols + j],
                matrix[(i + 2) * cols + j],
                matrix[(i + 3) * cols + j],
                matrix[(i + 4) * cols + j],
                matrix[(i + 5) * cols + j],
                matrix[(i + 6) * cols + j],
                matrix[(i + 7) * cols + j]
            );
            _mm256_storeu_ps(&result[j * rows + i], vec);
        }
    }
}

void transpose_matrix(float* matrix, float* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
}

void print_matrix(const char* name, float* matrix, int rows, int cols) {
    printf("%s: \n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void check_integrity(float* result1, float* result2, int rows, int cols) {
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            if (result1[i * rows + j] != result2[i * rows + j]) {
                printf("Error: Los resultados no coinciden en el índice [%d][%d]\n", j, i);
                return;
            }
        }
    }
    printf("Verificación de integridad pasada: Los resultados coinciden.\n");
}
