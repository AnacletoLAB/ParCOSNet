#include "timer.h"

Timer::Timer() {}

double Timer::duration() {
	std::chrono::duration<double, std::milli> fp_ms = end - start;
	return fp_ms.count();
}

void Timer::startTime() {
	start = std::chrono::high_resolution_clock::now();
}

void Timer::endTime() {
	end = std::chrono::high_resolution_clock::now();
}
