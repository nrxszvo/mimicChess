#include <string>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include "utils/utils.h"

struct Block {
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	long total_nano;
	int count;
	int reportFmt;
	Block(): total_nano(0), count(0), reportFmt(0) {};
	Block(int reportFmt): total_nano(0), count(0), reportFmt(reportFmt) {};
};

#ifdef PROFILE_ENABLE
class Profiler {
	float alpha;
	std::unordered_map<std::string,Block> blocks;
	std::vector<std::string> names;
public:
	Profiler(float alpha=0.9): alpha(alpha) {};
	void init(std::string name, int reportFmt=0) {
		this->names.push_back(name);
		this->blocks[name] = Block(reportFmt);	
	}
	inline void start(std::string name) {
		this->blocks[name].start = std::chrono::high_resolution_clock::now();	
	}
	inline void stop(std::string name) {
		Block& block = this->blocks[name];
		auto stop = std::chrono::high_resolution_clock::now();
		auto start = block.start;
		long nano = std::chrono::duration_cast<std::chrono::nanoseconds>(stop-start).count();
		block.total_nano += nano;
		block.count++;
	}	
	long getNano(std::string name) {
		return this->blocks[name].total_nano;
	}

	float getAverage(Block& block) {
		float avg = (float)block.total_nano/1e6/block.count;
		block.total_nano = 0;
		block.count = 0;
		return avg;
	}
	void report() {
		for (auto& name: this->names) {
			Block& block = this->blocks[name];
			std::string val;
			if (block.reportFmt == 0) {
				val = std::to_string(getAverage(block)) + " ms";
			} else {
				int ellapsed = int(block.total_nano/1e9);
				val = getEllapsedStr(ellapsed);
			}
			std::cout << name << ": " << val << std::endl;
		}
	}
};
#else
class Profiler {
public:
	void init(std::string name, int reportFmt=0) {
		return;
	}
	inline void start(std::string name) {
		return;
	}
	inline void stop(std::string name) {
		return;
	}
	long getNano(std::string name) {
		return 0;
	}
	float getAverage(std::string name) {
		return 0.0f;
	}
	void report() {
		return;
	}
};
#endif

extern Profiler profiler;
