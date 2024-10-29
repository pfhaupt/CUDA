// #define DEBUG
#define ELEM_TYPE int
#include "common.h"

#include "raylib.h"

#define WIDTH 3840
#define HEIGHT 2160

#define cast(x, typ) (*((typ*)&(x)))

typedef ELEM_TYPE gpuType;

__device__ int color(gpuType *info, int id) {
#if 1
    int gray = info[id] * 255;
    int c = (gray << 16) | (gray << 8) | gray;
    return (0xFF << 24) | c;
#else
#error "more colors"
#endif
}

__device__ int getNeighbors(gpuType *grid, int x, int y, int w, int h) {
    int cnt = 0;
    for (int dx = -1; dx < 2; dx++) {
        for (int dy = -1; dy < 2; dy++) {
            if (dx == 0 && dy == 0) continue;
            int _x = (x + dx);
            int _y = (y + dy);
            if (_x < 0 || _y < 0 || _x >= w || _y >= h) continue;
            int id = _y * w + _x;
            cnt += grid[id];
        }
    }

    return cnt;
}

__global__ void gol(int *pixels, gpuType *before, gpuType *after, int width, int height) {
    const int N = width * height;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x;
        id < N; id += blockDim.x * gridDim.x) {
        int _x = id % width;
        int _y = id / width;
        int cnt = getNeighbors(before, _x, _y, width, height);
        if (before[id] && (cnt == 2 || cnt == 3)) after[id] = 1;
        else if (before[id] == 0 && cnt == 3) after[id] = 1;
//        if (before[id] && (cnt == 1 || cnt == 3 || cnt == 5 || cnt == 7)) after[id] = 1;
//        else if (before[id] == 0 && (cnt == 1 || cnt == 3 || cnt == 5 || cnt == 7)) after[id] = 1;
        else after[id] = 0;
        pixels[id] = color(after, id);
    }
}

int main() {
    hostCalloc(pixels, WIDTH * HEIGHT, int);
    hostCalloc(first, WIDTH * HEIGHT, gpuType);
    deviceCalloc(d_pixels, WIDTH * HEIGHT, int);
    deviceCalloc(d_before, WIDTH * HEIGHT, gpuType);
    deviceCalloc(d_after, WIDTH * HEIGHT, gpuType);

    InitWindow(WIDTH, HEIGHT, "ligma");

    SetTraceLogLevel(LOG_WARNING);
    SetTargetFPS(60);
    SetConfigFlags(FLAG_FULLSCREEN_MODE);

    for (int i = 0; i < WIDTH * HEIGHT; i++) first[i] = (gpuType)((rand() * 12390123) % 2);
    copyHostToDevice(d_before, first, WIDTH * HEIGHT);

    invokeKernel(gol, 4096, 1024, d_pixels, d_before, d_after, WIDTH, HEIGHT);
    copyDeviceToHost(pixels, d_pixels, WIDTH * HEIGHT);
    Image img = {
        pixels,
        WIDTH,
        HEIGHT,
        1,
        PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
    };
    Texture texture = LoadTextureFromImage(img);

    float lastExecTime = 0;
    bool swapped = false;
    bool showTime = false;
    while (!WindowShouldClose()) {
        bool screenUpdated = true;
        if (IsKeyDown(KEY_SPACE)) {
            screenUpdated = true;
        }
        if (IsKeyReleased(KEY_T)) showTime = !showTime;
        if (screenUpdated) {
            long long int ns;
            TIME_AND_BIND(ns,
                if (swapped) {
                    invokeKernel(gol, 4096, 1024, d_pixels, d_before, d_after, WIDTH, HEIGHT);
                } else {
                    invokeKernel(gol, 4096, 1024, d_pixels, d_after, d_before, WIDTH, HEIGHT);
                }
                ENSURE(cudaStreamSynchronize(0), "Could not synchronize Stream")
            );
            lastExecTime = (float)ns / 1'000'000;
            copyDeviceToHost(pixels, d_pixels, WIDTH * HEIGHT);
            UnloadTexture(texture);
            texture = LoadTextureFromImage(img);
            swapped = !swapped;
        }
        BeginDrawing();
        DrawTexture(texture, 0, 0, WHITE);
        if (showTime)
            DrawText(TextFormat("Kernel took: %.6f ms", lastExecTime), WIDTH / 2, HEIGHT / 2, 50, BLUE);
        EndDrawing();
    }
    return 0;
}
