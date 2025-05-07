#include <queue>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <thread>
#include "parser.h"

struct Data {
	int pid;
	float progress;
	int gameId;
	int welo;
	int belo;
	int time;
	int inc;
	std::string info; 

	Data() {};
	Data(int pid, std::string info): pid(pid), info(info) {};
	Data(int pid, std::string info, int gameId): pid(pid), info(info), gameId(gameId) {};
	Data(int pid, float progress, int gid, int welo, int belo, int time, int inc, std::string info)
		: pid(pid), progress(progress), gameId(gid), welo(welo), belo(belo), time(time), inc(inc), info(info) {};
	Data(int pid, std::shared_ptr<Data> other)
		: pid(pid), progress(other->progress), gameId(other->gameId), welo(other->welo), belo(other->belo), time(other->time), inc(other->inc) {};
};

struct GameData: Data {
	GameData(): Data() {};
	GameData(int pid, std::string info) : Data(pid, info) {};
	GameData(int pid, std::string info, int gameId): Data(pid, info, gameId) {};
	GameData(int pid, float progress, int gid, int welo, int belo, int time, int inc, std::string moveStr, std::string info) 
		: Data(pid, progress, gid, welo, belo, time, inc, info), moveStr(moveStr) {};

	std::string moveStr;
};

struct MoveData: Data {
	MoveData(int pid, std::string info) : Data(pid, info) {};
	MoveData(int pid, std::string info, int gameId): Data(pid, info, gameId) {};
	MoveData(
			int pid, 
			std::shared_ptr<GameData> gd, 
			std::string mvs,
			std::string clk,
			std::string eval,
			uint8_t result
			) 
		: Data(pid, gd), mvs(mvs), clk(clk), eval(eval), result(result) {
		this->info = "GAME";
	};

	std::string mvs;
	std::string clk;
	std::string eval;
	uint8_t result;
};

class ParallelParser {
	std::queue<std::shared_ptr<GameData> > gamesQ;
	std::queue<std::shared_ptr<MoveData> > outputQ;
	std::mutex gamesMtx;
	std::mutex outputMtx;
	std::condition_variable gamesCv;
	std::condition_variable outputCv;
	int nReaders;
	int nMoveProcessors;
	int minSec;
	int maxSec;
	int maxInc;
	std::vector<std::shared_ptr<std::thread> > procThreads;
	std::vector<std::shared_ptr<std::thread> > gameThreads;
public:
	ParallelParser(int nReaders, int nMoveProcessors, int minSec, int maxSec, int maxInc);
	~ParallelParser();
	std::shared_ptr<ParserOutput> parse(std::string zst, std::string name, int printFreq=60);
};
