// #define DEBUG
#include "common.h"

#include "raylib.h"

#define WIDTH 3840
#define HEIGHT 2160

#define cast(x, typ) (*((typ*)&(x)))

typedef double gpuType;

// #define USE_DOUBLE_DEKKER


/* BEGIN DOUBLE IT LIKE DEKKER */
// Source from Creel yt channel videos:
//  https://www.youtube.com/watch?v=6OuqnaHHUG8
//  https://www.youtube.com/watch?v=5IL1LJ5noww
// adapted to go from float to double precision

typedef struct{
    float upper;
    float lower;
}DD;

// SCALE is 2^(mantissabits/2)+1 = 2^(24/2)+1 = 4097
const float SCALE = 4097;

// float -> DD
__device__ DD dd_split(float a){
    DD R;
    float p = a*SCALE;
    R.upper = (a-p)+p;
    R.lower = a-R.upper;
    return R;
}

// DD -> float (loss of precision)
__device__ float dd_join(DD a){
    return a.upper+a.lower; //TODO: is this really it?
}

//  DD + DD -> DD
__device__ DD dd_add(const DD a, const DD b){
    DD r;
    r.upper = a.upper+b.upper;
    r.lower = 0;

    if(abs(a.upper)>abs(b.upper)){
        r.lower = a.upper-r.upper+b.upper+b.lower+a.lower;
    }else{
        r.lower = b.upper-r.upper+a.upper+a.lower+b.lower;
    }

    DD c;
    c.upper = r.upper+r.lower;
    c.lower = r.upper-c.upper+r.lower;

    return c;
}

// DD - DD -> DD
__device__ DD dd_sub(const DD a, const DD b){
    DD r;
    r.upper = a.upper-b.upper;
    r.lower = 0;

    if(abs(a.upper)>abs(b.upper)){
        r.lower = a.upper-r.upper-b.upper-b.lower+a.lower;
    }else{
        r.lower = -b.upper-r.upper+a.upper+a.lower-b.lower;
    }

    DD c;
    c.upper = r.upper+r.lower;
    c.lower = r.upper-c.upper+r.lower;

    return c;
}

// float * float -> DD
__device__ DD dd_mul12(float a, float b){
    DD A = dd_split(a);
    DD B = dd_split(b);

    float p = A.upper*B.upper;
    float q = A.upper*B.lower+A.lower*B.upper;

    DD R;
    R.upper = p+q;
    R.lower = p-R.upper+q+A.lower*B.lower;

    return R;
}

// DD * DD -> DD
__device__ DD dd_mul(const DD a, const DD b){
    DD t = dd_mul12(a.upper,b.upper);
    float c = a.upper*b.lower+a.lower*b.upper + t.lower;

    DD r;
    r.upper = t.upper+c;
    r.lower = t.upper-r.upper+c;

    return r;
}

// DD / DD -> DD (a/b)
__device__ DD dd_div(const DD a, const DD b){
    DD u;
    u.upper = a.upper/b.upper;
    DD t = dd_mul12(u.upper,b.upper);

    /* float l = (a.upper-t.upper-t.lower+n.lower-u.upper*b.lower)/b.upper; */
    float l = ((((a.upper-t.upper)-t.lower)+a.lower)-u.upper*b.lower)/b.upper;

    DD r;
    r.upper = u.upper+l;
    r.lower = u.upper-r.upper+l;

    return r;
}


/* END OF DOUBLE IT LIKE DEKKER */



__device__ int compute_dd(gpuType *info, int id, int width, int height) {
    DD screenMiddleX = dd_split(info[0]);
    DD screenMiddleY = dd_split(info[1]);
    DD screenW = dd_split(info[2]);
    DD screenH = dd_split(info[3]);
    DD dd2 = dd_split(2.0);
    int maxIter = cast(info[4], int);
    DD screenX = dd_sub(screenMiddleX, dd_div(screenW , dd2));
    DD screenY = dd_sub(screenMiddleY, dd_div(screenH , dd2));
    int _x = id % width;
    int _y = id / width;
    DD ddwidth = dd_split(width);
    DD ddheight = dd_split(height);
    DD percX = dd_div(dd_split(_x), ddwidth);
    DD percY = dd_div(dd_split(_y), ddheight);
    DD posx, posy;
    if (width < height) {
        posx = dd_add(screenX , dd_mul(percX,screenW));
        posy = dd_mul(dd_add(screenY , dd_mul(percY,screenH)) , dd_div(ddheight,ddwidth));
    } else {
        posx = dd_mul(dd_add(screenX , dd_mul(percX,screenW)) , dd_div(ddwidth,ddheight));
        posy = dd_add(screenY , dd_mul(percY,screenH));
    }
    int i = 0;
    DD zx = posx;
    DD zy = posy;
    DD zrsqr = dd_mul(zx,zx);
    DD zisqr = dd_mul(zy,zy);
    DD a = dd_add(zrsqr,zisqr);
    while (i < maxIter && dd_join(a) < 256) {
        a = dd_add(zrsqr,zisqr);
        zy = dd_mul(zx,zy);
        zy = dd_add(zy,zy);
        zy = dd_add(zy,posy);
        zx = dd_add(dd_sub(zrsqr,zisqr),posx);
        zrsqr = dd_mul(zx,zx);
        zisqr = dd_mul(zy,zy);
        i++;
    }
    if (i == maxIter) return 0;
    return i;
}



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
#ifndef USE_DOUBLE_DEKKER
        int iter = compute(screen, id, width, height);
#else
        int iter = compute_dd(screen, id, width, height);
#endif
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
    ENSURE(cudaStreamSynchronize(0), "First pass failed");
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
