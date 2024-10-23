#include <assert.h>
#include <stdio.h>

#define DEBUG
#include "common.h"

#define SIZE 500'000'000

__global__ void add(int *c, int *b, int *a, const int N) {
    for ( int id = blockIdx.x * blockDim.x + threadIdx.x;
        id < N; id += blockDim.x * gridDim.x) {
        c[id] = b[id] + a[id];
    }
}

int main() {
    hostCalloc(a_host, SIZE, int);
    hostCalloc(b_host, SIZE, int);
    hostCalloc(c_host, SIZE, int);
    hostCalloc(d_host, SIZE, int);
    for (int i = 0; i < SIZE; i++) {
        a_host[i] = i;
        b_host[i] = i;
        d_host[i] = 2 * i;
    }
    deviceCalloc(a_device, SIZE, int);
    deviceCalloc(b_device, SIZE, int);
    deviceCalloc(c_device, SIZE, int);

    copyHostToDevice(a_device, a_host, SIZE);
    copyHostToDevice(b_device, b_host, SIZE);

    TIME(
        invokeKernel(add, 4096, dim3(32, 2, 2), c_device, b_device, a_device, SIZE);
        ENSURE(cudaStreamSynchronize(0), "Could not synchronize with device");
        , "GPU addition"
    )

    copyDeviceToHost(c_host, c_device, SIZE);

    verifyMemory(c_host, d_host, SIZE);

    freeHost();
    freeDevice();
    printf("Success!\n");
    return 0;
}
