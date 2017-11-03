#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define SIGN(x) (x < 0 ? LO : HI)
#define SIGNTH(x) (x < 0 ? negState : posState)
#define SIGNTHLAMBDA(x) (x < 0 ? nS : pS)

// draw a random float in [a, b)
__inline float randf(float a, float b) {
	float r = (float) ((float) rand() / (float) RAND_MAX);
	return r * (b - a) + a;
}
