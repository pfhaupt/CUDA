// #define DEBUG
#include "common.h"

#include "raylib.h"

#define WIDTH 3840
#define HEIGHT 2160

#define BALL_COUNT 5
#define MIN_RADIUS 0.1f
#define MAX_RADIUS 0.1f

#define GRAVITY 0.0002f
#define DAMPING 0.99

#define cast(x, typ) (*((typ*)&(x)))

typedef struct {
    int id;
    float px;
    float py;
    float vx;
    float vy;
    float nextPx;
    float nextPy;
    float nextVx;
    float nextVy;
    float ax;
    float ay;
    float radius;
    float mass;
    Color color;
} Ball;

__device__ bool ballsCollided(Ball *first, Ball *second) {
    float fx = first->px;
    float fy = first->py;
    float fr = first->radius;
    float sx = second->px;
    float sy = second->py;
    float sr = second->radius;
    float dx = fx - sx;
    float dy = fy - sy;
    float dr = fr + sr;
    return dx * dx + dy * dy <= dr * dr;
}

__global__ void updateBalls(Ball *balls, int ballCount, float aspectRatio) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x;
        id < ballCount; id += blockDim.x * gridDim.x) {
        Ball *ball = &balls[id];
        float r = ball->radius;
        float nVx = ball->vx;
        float nVy = ball->vy;
        for (int i = 0; i < ballCount; i++) {
            Ball *other = &balls[i];
            if (ball->id == other->id) continue;
            if (ballsCollided(ball, other)) {
                float m1 = ball->mass;
                float m2 = other->mass;
                float p1x = ball->px;
                float p1y = ball->py;
                float p2x = other->px;
                float p2y = other->py;
                float v1x = nVx;
                float v1y = nVy;
                float v2x = other->vx;
                float v2y = other->vy;
                float m1x = p1x + v1x * 0.01f;
                float m1y = p1y + v1y * 0.01f;
                float m2x = p2x + v2x * 0.01f;
                float m2y = p2y + v2y * 0.01f;
                float dc = (p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y);
                float dn = (m1x - m2x) * (m1x - m2x) + (m1y - m2y) * (m1y - m2y);
                float t = (v1x - v2x) * (p1x - p2x) + (v1y - v2y) * (p1y - p2y);
                float d = (p1x - p2x) * (p1x - p2x) + (p1y - p2y) * (p1y - p2y);
                float mm = 2 * m2 / (m1 + m2);
                if (dn < dc) {
                    nVx = (v1x - mm * t / d * (p1x - p2x));
                    nVy = (v1y - mm * t / d * (p1y - p2y));
                    unsigned int c = 0xFF00FF00;
                    ball->color = cast(c, Color);
                } else {
                    unsigned int c = 0xFF0000FF;
                    ball->color = cast(c, Color);
                }
            }
        }
        nVx = nVx + ball->ax;
        nVy = nVy + ball->ay;
        float nx = ball->px + nVx;
        float ny = ball->py + nVy;
        if (nx <= r) {
            nx += r - nx;
            nVx *= -1;
        }
        if (nx >= 1 - r) {
            nx += (1 - r) - nx;
            nVx *= -1;
        }
        if (ny <= r) {
            ny += r - ny;
            nVy *= -1;
        }
        if (ny >= 1 / aspectRatio - r) {
            ny += (1 / aspectRatio - r) - ny;
            nVy *= -1;
        }
        ball->nextPx = nx;
        ball->nextPy = ny;
        ball->nextVx = nVx * DAMPING;
        ball->nextVy = nVy * DAMPING;
    }
}

__global__ void updateBalls2(Ball *balls, int ballCount, float aspectRatio) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x;
        id < ballCount; id += blockDim.x * gridDim.x) {
        Ball *ball = &balls[id];
        ball->px = ball->nextPx;
        ball->py = ball->nextPy;
        ball->vx = ball->nextVx;
        ball->vy = ball->nextVy;
    }
}

float randFloat() {
    unsigned int r = rand() * 1289398213;
    return (float)r / (float)UINT32_MAX;
}

#define randByte ((unsigned int)(rand() % 256))

bool hostBallsCollided(Ball *first, Ball *second) {
    float fx = first->px;
    float fy = first->py;
    float fr = first->radius;
    float sx = second->px;
    float sy = second->py;
    float sr = second->radius;
    float dx = fx - sx;
    float dy = fy - sy;
    float dr = fr + sr;
    return dx * dx + dy * dy <= dr * dr;
}

void initializeBalls(Ball *d_balls, int ballCount, float aspectRatio) {
    hostCalloc(h_balls, ballCount, Ball);
    for (int i = 0; i < ballCount; i++) {
        Color color = CLITERAL(Color) {
            (unsigned char)0xFF,
            (unsigned char)0xFF,
            (unsigned char)0xFF,
            (unsigned char)0xFF
        };
        bool succ;
        Ball newball;
        int failsafe = 0;
        do {
            float r = MIN_RADIUS + randFloat() * (MAX_RADIUS - MIN_RADIUS);
            newball = Ball {
                i,
                r + (randFloat() * (1 - 2 * r)), // pos x
                r + (randFloat() * (1 / aspectRatio - 2 * r)), // pos y
                0, // vel x
                0, // vel y
                0, // new pos x
                0, // new pos y
                0, // new vel x
                0, // new vel y
                0, // acc x
                GRAVITY, // acc y
                r, // radius
                0.1, // mass
                color,
            };
            succ = true;
            for (int j = 0; j < i; j++) {
                Ball *other = &h_balls[j];
                if (hostBallsCollided(&newball, other)) {
                    succ = false;
                    break;
                }
            }
            if (failsafe++ >= 1000) {
                printf("could not initialize ball %d\n", i);
                exit(1);
            }
        } while(!succ);
        h_balls[i] = newball;
    }
    copyHostToDevice(d_balls, h_balls, ballCount);
    free(h_balls);
}

int main() {
    srand(time(NULL));
    hostCalloc(h_balls, BALL_COUNT, Ball);
    deviceCalloc(d_balls, BALL_COUNT, Ball);

    const float aspectRatio = (float)WIDTH / (float)HEIGHT;

    InitWindow(WIDTH, HEIGHT, "ligma");

    SetTraceLogLevel(LOG_WARNING);
    SetTargetFPS(60);
    SetConfigFlags(FLAG_FULLSCREEN_MODE);

    initializeBalls(d_balls, BALL_COUNT, aspectRatio);

    float lastExecTime = 0;
    bool showText = true;
    while (!WindowShouldClose()) {
        if (IsKeyReleased(KEY_TAB)) showText = !showText;
        if (IsKeyReleased(KEY_SPACE)) initializeBalls(d_balls, BALL_COUNT, aspectRatio);
        {
            long long int ns;
            TIME_AND_BIND(ns,
                invokeKernel(updateBalls, 4096, 512, d_balls, BALL_COUNT, aspectRatio);
                invokeKernel(updateBalls2, 4096, 512, d_balls, BALL_COUNT, aspectRatio);
                ENSURE(cudaStreamSynchronize(0), "Could not synchronize Stream")
            );
            lastExecTime = (float)ns / 1'000'000;
            copyDeviceToHost(h_balls, d_balls, BALL_COUNT);
        }
        BeginDrawing();
        ClearBackground(BLACK);
        long long int drawTime;
        TIME_AND_BIND(drawTime,
            for (int i = 0; i < BALL_COUNT; i++) {
                Ball *ball = &h_balls[i];
                DrawCircle(ball->px * WIDTH, ball->py * WIDTH, ball->radius * WIDTH, ball->color);
            }
        );
        if (showText) {
            DrawText(TextFormat("Ball count: %d", BALL_COUNT), 50, 50, 50, BLUE);
            DrawText(TextFormat("Kernel took: %.6f ms", lastExecTime), 50, 100, 50, BLUE);
            DrawText(TextFormat("Drawing took: %.6f ms", (float)drawTime / 1'000'000), 50, 150, 50, BLUE);
        }
        EndDrawing();
    }
    return 0;
}
