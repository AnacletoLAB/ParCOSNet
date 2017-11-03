#pragma once
#include <chrono>
#include <iostream>

class Timer {
public:
	Timer();

	void startTime();
	void endTime();
	double duration();

protected:
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
};
