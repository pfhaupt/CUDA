// #define DEBUG
#define ELEM_TYPE float
#include "common.h"

#include "raylib.h"

#define WIDTH 3840
#define HEIGHT 2160

#define AA_LEVEL 2
#define AA_PER_PIXEL (AA_LEVEL * AA_LEVEL)

#define cast(x, typ) (*((typ*)&(x)))

#define map(val, from, to, low, high) (low + (val - from) / (to - from) * (high - low))

typedef float gpuType;

__device__ int compute(gpuType posx, gpuType posy, gpuType posz, int maxIter) {
    gpuType zx = posx;
    gpuType zy = posy;
    gpuType zz = posz;
    int i = 0;
    const int n = 8;
    gpuType sumSq = zx * zx + zy * zy + zz * zz;
    while (i < maxIter && sumSq < 4) {
        gpuType r = sqrt(sumSq);
        gpuType rn = __powf(r, n);
        gpuType phi = atan2f(zy, zx);
        gpuType theta = acosf(zz / r);
        gpuType sinTheta = __sinf(n * theta);
        gpuType cosTheta = __cosf(n * theta);
        gpuType sinPhi = __sinf(n * phi);
        gpuType cosPhi = __cosf(n * phi);
        gpuType nx = rn * sinTheta * cosPhi;
        gpuType ny = rn * sinTheta * sinPhi;
        gpuType nz = rn * cosTheta;
        zx = nx + posx;
        zy = ny + posy;
        zz = nz + posz;
        sumSq = zx * zx + zy * zy + zz * zz;
        i++;
    }
    return i;
}

__device__ void calculatePixel(gpuType *info, int id, float *paramIter, float *paramDepth) {
    gpuType Ax = info[0];
    gpuType Ay = info[1];
    gpuType Az = info[2];
    gpuType stepIx = info[3];
    gpuType stepIy = info[4];
    gpuType stepIz = info[5];
    gpuType stepJx = info[6];
    gpuType stepJy = info[7];
    gpuType stepJz = info[8];
    gpuType stepKx = info[9];
    gpuType stepKy = info[10];
    gpuType stepKz = info[11];
    int maxIter = cast(info[12], int);
    int maxDepth = cast(info[13], int);
    int _x = id % WIDTH;
    int _y = id / WIDTH;
    float totalIter = 0;
    float totalDepth = 0;
    for (int aax = 0; aax < AA_LEVEL; aax++) {
        float nx = (float)_x + (float)aax / (float)AA_LEVEL;
        for (int aay = 0; aay < AA_LEVEL; aay++) {
            float ny = (float)_y + (float)aay / (float)AA_LEVEL;
            gpuType posx = Ax - stepIx * nx - stepJx * ny;
            gpuType posy = Ay - stepIy * nx - stepJy * ny;
            gpuType posz = Az - stepIz * nx - stepJz * ny;
            int i = 0;
            int r = maxDepth;
            for (int depth = 0; depth < maxDepth; depth++) {
                i = compute(posx, posy, posz, maxIter);
                if (i == maxIter) {
                    r = depth;
                    break;
                }
                posx += stepKx;
                posy += stepKy;
                posz += stepKz;
            }
            totalIter += (float)i;
            totalDepth += (float)r;
        }
    }
    *paramIter = totalIter;
    *paramDepth = totalDepth;
}

__device__ int color(gpuType *info, float totalIter, float totalDepth) {
    int maxIter = cast(info[12], int);
    int maxDepth = cast(info[13], int);
    float scaledIter = totalIter / AA_PER_PIXEL;
    float scaledDepth = totalDepth / AA_PER_PIXEL;
    int color;
    if (maxDepth == 1)
    	color = (int)map(scaledIter, 0, (float)maxIter, 0, 255);
    else
        color = (int)map(scaledDepth, 0, (float)maxDepth, 0, 255);
    int gray = color << 16 | color << 8 | color;
    return 0xFF << 24 | gray;
//     float _iter = (float)iter;
//     float lIter = __log2f(_iter);
//     float mIter = __log2f((float)maxIter);
//     float _color = map(lIter, 0, mIter, 0, 255);
//     int color = (int)_color;
}

__global__ void mandel(gpuType *screen, int *pixels) {
    const int N = WIDTH * HEIGHT;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x;
        id < N; id += blockDim.x * gridDim.x) {
        float totalIter, totalDepth;
        calculatePixel(screen, id, &totalIter, &totalDepth);
        int col = color(screen, totalIter, totalDepth);
        pixels[id] = col;
    }
}

void resetInfo(gpuType *info) {
    int defaultIter = 100;
    int defaultDepth = 100;
    // Top Left
    info[0] = -2;
    info[1] = -2;
    info[2] = 0;
    // Step X
    info[3] = 4.0 / (float)WIDTH;
    info[4] = 0;
    info[5] = 0;
    // Step Y
    info[6] = 0;
    info[7] = 4.0 / (float)HEIGHT;
    info[8] = 0;
    // Step Z
    info[9] = 0;
    info[10] = 0;
    info[11] = 0;
    info[12] = cast(defaultIter, gpuType);
    info[13] = cast(defaultDepth, gpuType);
}

Vector3 Vector3Sub(Vector3 v1, Vector3 v2)
{
    Vector3 result = { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };

    return result;
}
Vector3 Vector3Add(Vector3 v1, Vector3 v2)
{
    Vector3 result = { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };

    return result;
}

Vector3 Vector3Scale(Vector3 v, float scalar)
{
    Vector3 result = { v.x*scalar, v.y*scalar, v.z*scalar };

    return result;
}

float Vector3Length(const Vector3 v)
{
    float result = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);

    return result;
}

int main() {
    hostCalloc(pixels, WIDTH * HEIGHT, int);
    deviceCalloc(d_pixels, WIDTH * HEIGHT, int);

    InitWindow(WIDTH, HEIGHT, "ligma");

    SetTraceLogLevel(LOG_WARNING);
    SetTargetFPS(60);
    SetConfigFlags(FLAG_FULLSCREEN_MODE);

    gpuType info[14] = {0};
    resetInfo(info);
    int infoSize = sizeof(info) / sizeof(info[0]);
    deviceCalloc(d_screen, infoSize, gpuType);
    copyHostToDevice(d_screen, info, infoSize);

    invokeKernel(mandel, 4096, 512, d_screen, d_pixels);
    copyDeviceToHost(pixels, d_pixels, WIDTH * HEIGHT);
    Image img = {
        pixels,
        WIDTH,
        HEIGHT,
        1,
        PIXELFORMAT_UNCOMPRESSED_R8G8B8A8,
    };
    Texture texture = LoadTextureFromImage(img);

    Vector3 position = {0};
    Vector3 ei = {1, 0, 0};
    Vector3 ej = {0, 1, 0};
    Vector3 ek = {0, 0, 1};
    float dimScale = 0.01;
    float velScale = 0.01;
    float lastExecTime = 0;
    bool firstFrame = true;
    while (!WindowShouldClose()) {
        float alpha = 0.01;
        bool screenUpdated = firstFrame;
        firstFrame = false;
        if (IsKeyDown(KEY_W)) {
            Vector3 ej1 = Vector3Add(Vector3Scale(ej, cos(alpha)), Vector3Scale(ek, sin(alpha)));
            Vector3 ek1 = Vector3Add(Vector3Scale(ek, cos(alpha)), Vector3Scale(ej, -sin(alpha)));
            ej = ej1;
            ek = ek1;
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_S)) {
            alpha = -alpha;
            Vector3 ej1 = Vector3Add(Vector3Scale(ej, cos(alpha)), Vector3Scale(ek, sin(alpha)));
            Vector3 ek1 = Vector3Add(Vector3Scale(ek, cos(alpha)), Vector3Scale(ej, -sin(alpha)));
            ej = ej1;
            ek = ek1;
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_A)) {
            Vector3 ei1 = Vector3Add(Vector3Scale(ei, cos(alpha)), Vector3Scale(ek, -sin(alpha)));
            Vector3 ek1 = Vector3Add(Vector3Scale(ek, cos(alpha)), Vector3Scale(ei, sin(alpha)));
            ei = ei1;
            ek = ek1;
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_D)) {
            alpha = -alpha;
            Vector3 ei1 = Vector3Add(Vector3Scale(ei, cos(alpha)), Vector3Scale(ek, -sin(alpha)));
            Vector3 ek1 = Vector3Add(Vector3Scale(ek, cos(alpha)), Vector3Scale(ei, sin(alpha)));
            ei = ei1;
            ek = ek1;
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_E)) {
            Vector3 ei1 = Vector3Add(Vector3Scale(ei, cos(alpha)), Vector3Scale(ej, sin(alpha)));
            Vector3 ej1 = Vector3Add(Vector3Scale(ej, cos(alpha)), Vector3Scale(ei, -sin(alpha)));
            ei = ei1;
            ej = ej1;
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_Q)) {
            alpha = -alpha;
            Vector3 ei1 = Vector3Add(Vector3Scale(ei, cos(alpha)), Vector3Scale(ej, sin(alpha)));
            Vector3 ej1 = Vector3Add(Vector3Scale(ej, cos(alpha)), Vector3Scale(ei, -sin(alpha)));
            ei = ei1;
            ej = ej1;
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_UP)) {
            position = Vector3Add(position, Vector3Scale(ek, velScale));
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_DOWN)) {
            position = Vector3Sub(position, Vector3Scale(ek, velScale));
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_Z)) {
            dimScale *= 1.01;
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_X)) {
            dimScale /= 1.01;
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_SPACE)) {
            position = {0};
            ei = {1, 0, 0};
            ej = {0, 1, 0};
            ek = {0, 0, 1};
            dimScale = 0.01;
            velScale = 0.01;
            resetInfo(info);
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_COMMA)) {
            int it = cast(info[12], int);
            int nw = (int)((gpuType)it / 1.01);
            if (it == nw) it--;
            else it = nw;
            if (it < 10) {
                it = 10;
                TraceLog(LOG_WARNING, "Minimum iteration count is capped to 10.");
            } else {
                info[12] = cast(it, gpuType);
                screenUpdated = true;
            }
        }
        if (IsKeyDown(KEY_PERIOD)) {
            int it = cast(info[12], int);
            int nw = (int)((gpuType)it * 1.01);
            if (it == nw) it++;
            else it = nw;
            info[12] = cast(it, gpuType);
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_R)) {
            int depth = cast(info[13], int);
            int nw = (int)((gpuType)depth / 1.01);
            if (depth == nw) depth--;
            else depth = nw;
            if (depth < 1) {
                depth = 1;
            } else {
                info[13] = cast(depth, gpuType);
                screenUpdated = true;
            }
        }
        if (IsKeyDown(KEY_T)) {
            int depth = cast(info[13], int);
            int nw = (int)((gpuType)depth * 1.01);
            if (depth == nw) depth++;
            else depth = nw;
            info[13] = cast(depth, gpuType);
            screenUpdated = true;
        }
        if (IsKeyDown(KEY_Y)) {
            int depth = 1;
            info[13] = cast(depth, gpuType);
            screenUpdated = true;
        }
        if (screenUpdated) {
            // Step X
            info[3] = ei.x * dimScale;
            info[4] = ei.y * dimScale;
            info[5] = ei.z * dimScale;
            // Step Y
            info[6] = ej.x * dimScale;
            info[7] = ej.y * dimScale;
            info[8] = ej.z * dimScale;
            // Step Z
            info[9] = ek.x * dimScale;
            info[10] = ek.y * dimScale;
            info[11] = ek.z * dimScale;
            // Top Left
            info[0] = position.x + info[3] * WIDTH / 2 + info[6] * HEIGHT / 2;
            info[1] = position.y + info[4] * WIDTH / 2 + info[7] * HEIGHT / 2;
            info[2] = position.z + info[5] * WIDTH / 2 + info[8] * HEIGHT / 2;
            copyHostToDevice(d_screen, info, infoSize);
            long long int ns;
            TIME_AND_BIND(ns,
                invokeKernel(mandel, 4096, 512, d_screen, d_pixels);
                ENSURE(cudaStreamSynchronize(0), "Could not synchronize Stream")
            );
            lastExecTime = (float)ns / 1'000'000;
            copyDeviceToHost(pixels, d_pixels, WIDTH * HEIGHT);
            UnloadTexture(texture);
            texture = LoadTextureFromImage(img);
        }
        BeginDrawing();
        DrawTexture(texture, 0, 0, WHITE);
        DrawText(TextFormat("Center: (%.12f, %.12f, %.12f)", position.x, position.y, position.z), 50, 50, 50, BLUE);
        DrawText(TextFormat("ei: (%.12f, %.12f, %.12f, %.12f)", ei.x, ei.y, ei.z, Vector3Length(ei)), 50, 100, 50, BLUE);
        DrawText(TextFormat("ej: (%.12f, %.12f, %.12f, %.12f)", ej.x, ej.y, ej.z, Vector3Length(ej)), 50, 150, 50, BLUE);
        DrawText(TextFormat("ek: (%.12f, %.12f, %.12f, %.12f)", ek.x, ek.y, ek.z, Vector3Length(ek)), 50, 200, 50, BLUE);
        DrawText(TextFormat("Kernel took: %.6f ms", lastExecTime), 50, 250, 50, BLUE);
        DrawText(TextFormat("Iterations: %d", cast(info[12], int)), 50, 300, 50, BLUE);
        DrawText(TextFormat("Depth: %d", cast(info[13], int)), 50, 350, 50, BLUE);
        EndDrawing();
    }

    freeHost();
    freeDevice();
    return 0;
}
