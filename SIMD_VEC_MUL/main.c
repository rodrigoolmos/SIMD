#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h> // AVX-512
#include <time.h>

#define VECTOR_SIZE 512*32 // Definir tamaño del vector

static float a[VECTOR_SIZE];
static float b[VECTOR_SIZE];
static float resultSMD[VECTOR_SIZE];
static float result[VECTOR_SIZE];

void fill_random(float* array, int size);
void multiply_vectors_SMD(float* a, float* b, float* result, int size);
void multiply_vectors(float* a, float* b, float* result, int size);
void print_vector(const char* name, float* array, int size);
void check_integrity(float* result1, float* result2, int size);

int main() {
    srand(time(NULL));

    // Llenar vectores con valores aleatorios
    fill_random(a, VECTOR_SIZE);
    fill_random(b, VECTOR_SIZE);
    while(1){
    // Medir tiempo de ejecución y realizar la multiplicación SIMD
    clock_t start = clock();
    multiply_vectors_SMD(a, b, resultSMD, VECTOR_SIZE);
    clock_t end = clock();
    double time_smd = (double)(end - start) / CLOCKS_PER_SEC;

    // Medir tiempo de ejecución y realizar la multiplicación convencional
    start = clock();
    multiply_vectors(a, b, result, VECTOR_SIZE);
    end = clock();
    double time_standard = (double)(end - start) / CLOCKS_PER_SEC;

    // Verificar la integridad de los datos
    check_integrity(resultSMD, result, VECTOR_SIZE);

    // Imprimir tiempos de ejecución
    printf("Tiempo de ejecución (SMD): %f segundos\n", time_smd);
    printf("Tiempo de ejecución (Standard): %f segundos\n", time_standard);
    }
    return 0;
}

void fill_random(float* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (float)rand() / RAND_MAX * 100.0f; // Valores aleatorios entre 0 y 100
    }
}

void multiply_vectors_SMD(float* a, float* b, float* result, int size) {
    int aligned_size = (size + 15) / 16 * 16; // Alinear tamaño a múltiplo de 16

    for (int i = 0; i < aligned_size; i += 16) {
        __m512 vecA = _mm512_loadu_ps(&a[i]);
        __m512 vecB = _mm512_loadu_ps(&b[i]);
        __m512 vecResult = _mm512_mul_ps(vecA, vecB);
        _mm512_storeu_ps(&result[i], vecResult);
    }
}

void multiply_vectors(float* a, float* b, float* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] * b[i];
    }
}

void print_vector(const char* name, float* array, int size) {
    printf("%s: \n", name);
    for (int i = 0; i < size; i++) {
        printf("%f ", array[i]);
    }
    printf("\n");
}

void check_integrity(float* result1, float* result2, int size) {
    for (int i = 0; i < size; i++) {
        if (result1[i] != result2[i]) {
            printf("Error: Los resultados no coinciden en el índice %d\n", i);
            return;
        }
    }
    printf("Verificación de integridad pasada: Los resultados coinciden.\n");
}
