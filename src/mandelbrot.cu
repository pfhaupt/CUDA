// #define DEBUG
#include "common.h"

#include "raylib.h"

#define WIDTH 3840
#define HEIGHT 2160

#define cast(x, typ) (*((typ*)&(x)))

typedef double gpuType;

__device__ int compute(gpuType *info, int id, int width, int height) {
    gpuType screenMiddleX = info[0];
    gpuType screenMiddleY = info[1];
    gpuType screenW = info[2];
    gpuType screenH = info[3];
    int maxIter = cast(info[4], int);
    gpuType screenX = screenMiddleX - screenW / 2;
    gpuType screenY = screenMiddleY - screenH / 2;
    int _x = id % width;
    int _y = id / width;
    gpuType percX = (gpuType)_x / (gpuType)width;
    gpuType percY = (gpuType)_y / (gpuType)height;
    gpuType posx, posy;
    if (width < height) {
        posx = screenX + percX * screenW;
        posy = (screenY + percY * screenH) * (gpuType)height / (gpuType)width;
    } else {
        posx = (screenX + percX * screenW) * (gpuType)width / (gpuType)height;
        posy = screenY + percY * screenH;
    }
    int i = 0;
    gpuType zx = posx;
    gpuType zy = posy;
    gpuType zrsqr = zx * zx;
    gpuType zisqr = zy * zy;
    gpuType a = zrsqr + zisqr;
    while (i < maxIter && a < 256) {
        a = zrsqr + zisqr;
        zy = zx * zy;
        zy += zy;
        zy += posy;
        zx = zrsqr - zisqr + posx;
        zrsqr = zx * zx;
        zisqr = zy * zy;
        i++;
    }
    if (i == maxIter) return 0;
    return i;
}

__device__ int color(gpuType *info, int iter) {
    long maxIter = cast(info[4], long);
#if 1
    return 0xFF << 24 | (int)(((gpuType)iter / maxIter)*255.0);
#else
#error "more colors"
#endif
}

__global__ void mandel(gpuType *screen, int *pixels, int width, int height) {
    const int N = width * height;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x;
        id < N; id += blockDim.x * gridDim.x) {
        int iter = compute(screen, id, width, height);
        int col = color(screen, iter);
        pixels[id] = col;
    }
}

void resetInfo(gpuType *info) {
    int defaultIter = 100;
    info[0] = 0;
    info[1] = 0;
    info[2] = 4;
    info[3] = 4;
    info[4] = cast(defaultIter, gpuType);
}

int main() {
    hostCalloc(pixels, WIDTH * HEIGHT, int);
    deviceCalloc(d_pixels, WIDTH * HEIGHT, int);

    InitWindow(WIDTH, HEIGHT, "ligma");

    SetTraceLogLevel(LOG_WARNING);
    SetTargetFPS(60);
    SetConfigFlags(FLAG_FULLSCREEN_MODE);

    gpuType info[5] = {0};
    resetInfo(info);
    int infoSize = sizeof(info) / sizeof(info[0]);
    deviceCalloc(d_screen, infoSize, gpuType);
    copyHostToDevice(d_screen, info, infoSize);

    invokeKernel(mandel, 4096, 1024, d_screen, d_pixels, WIDTH, HEIGHT);
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
    while (!WindowShouldClose()) {
        bool screenUpdated = false;
        if (IsKeyDown(KEY_S)) {
            SaveFileData("./assets/info.bin", info, sizeof(info));
        }
        if (IsKeyDown(KEY_L)) {
            int size;
            unsigned char *data = LoadFileData("./assets/info.bin", &size);
            if (data) {
                if (size == sizeof(info)) {
                    memcpy(info, data, sizeof(info));
                    screenUpdated = true;
                }
            }
        }
        if (IsKeyDown(KEY_SPACE)) {
            resetInfo(info);
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_RIGHT)) {
            info[0] += info[2] * 0.01;
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_LEFT)) {
            info[0] -= info[2] * 0.01;
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_UP)) {
            info[1] -= info[3] * 0.01;
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_DOWN)) {
            info[1] += info[3] * 0.01;
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_COMMA)) {
            int it = cast(info[4], int);
            int nw = (int)((gpuType)it / 1.01);
            if (it == nw) it--;
            else it = nw;
            if (it < 10) {
                it = 10;
                TraceLog(LOG_WARNING, "Minimum iteration count is capped to 10.");
            } else {
                info[4] = cast(it, gpuType);
                screenUpdated = true;
            }
        }
        if (IsKeyDown(KEY_PERIOD)) {
            int it = cast(info[4], int);
            int nw = (int)((gpuType)it * 1.01);
            if (it == nw) it++;
            else it = nw;
            info[4] = cast(it, gpuType);
            screenUpdated = true;
        }
        int mouseScroll = GetMouseWheelMove();
        if (mouseScroll != 0) {
            Vector2 mousePos = GetMousePosition();
            gpuType stepX = info[2] / WIDTH;
            gpuType stepY = info[3] / HEIGHT;
            gpuType i = mousePos.x;
            gpuType j = mousePos.y;
            gpuType x = info[0];
            gpuType y = info[1];
            gpuType nx, ny;
            if (mouseScroll < 0) {
                gpuType zoom = 1 + (0.05 * (gpuType)-mouseScroll);
                nx = x + (i - WIDTH / 2) * stepX * (1 / zoom - 1);
                ny = y + (j - HEIGHT / 2) * stepY * (1 / zoom - 1);
                info[2] *= zoom;
                info[3] *= zoom;
            } else {
                gpuType zoom = 1 + (0.05 * (gpuType)mouseScroll);
                nx = x + (i - WIDTH / 2) * stepX * (1 - 1 / zoom);
                ny = y + (j - HEIGHT / 2) * stepY * (1 - 1 / zoom);
                info[2] /= zoom;
                info[3] /= zoom;
            }
            info[0] = nx;
            info[1] = ny;
            screenUpdated = true;
        }
        if (screenUpdated) {
            copyHostToDevice(d_screen, info, infoSize);
            long long int ns;
            TIME_AND_BIND(ns,
                invokeKernel(mandel, 4096, 512, d_screen, d_pixels, WIDTH, HEIGHT);
                ENSURE(cudaStreamSynchronize(0), "Could not synchronize Stream")
            );
            lastExecTime = (float)ns / 1'000'000;
            copyDeviceToHost(pixels, d_pixels, WIDTH * HEIGHT);
            UnloadTexture(texture);
            texture = LoadTextureFromImage(img);
        }
        BeginDrawing();
        DrawTexture(texture, 0, 0, WHITE);
        DrawText(TextFormat("Center: (%.12f, %.12f)", info[0], info[1]), 50, 50, 50, BLUE);
        DrawText(TextFormat("Window: (%.12f, %.12f)", info[2], info[3]), 50, 100, 50, BLUE);
        DrawText(TextFormat("Iterations: %d", cast(info[4], int)), 50, 150, 50, BLUE);
        DrawText(TextFormat("Kernel took: %.6f ms", lastExecTime), 50, 200, 50, BLUE);
        EndDrawing();
    }
    return 0;
}
