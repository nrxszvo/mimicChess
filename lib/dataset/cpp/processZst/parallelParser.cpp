#include "parallelParser.h"
#include "lib/decompress.h"
#include "lib/parseMoves.h"
#include "parser.h"
#include "utils/utils.h"
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <unordered_set>

struct GameState {
	std::queue<std::shared_ptr<GameData> >* gamesQ;
	std::mutex* gamesMtx;
	std::condition_variable* gamesCv;
	int pid;
	GameState(
			std::queue<std::shared_ptr<GameData> >* gamesQ, 
			std::mutex* gamesMtx, 
			std::condition_variable* gamesCv) 
		: pid(-1), gamesQ(gamesQ), gamesMtx(gamesMtx), gamesCv(gamesCv) {};

};

void loadGamesZst(GameState gs, std::string zst, size_t frameStart, size_t frameEnd, int nMoveProcessors, int minSec, int maxSec, int maxInc) {
	int gameId = 0;
	int gamestart = 0;
	int lineno = 0;
	PgnProcessor processor(minSec, maxSec, maxInc);
	DecompressStream decompressor(zst, frameStart, frameEnd);
	while(decompressor.decompressFrame() != 0) {
		std::vector<std::string> lines;
		decompressor.getLines(lines);
		for (auto line: lines) {
			lineno++;
			std::string code = processor.processLine(line);
			if (code == "COMPLETE") {
				{
					std::lock_guard<std::mutex> lock(*gs.gamesMtx);
					gs.gamesQ->push(std::make_shared<GameData>(
								gs.pid,
								decompressor.getProgress(),
								gameId, 
								processor.getWelo(), 
								processor.getBelo(), 
								processor.getTime(),
								processor.getInc(),
								processor.getMoveStr(),
								zst + ":" + std::to_string(gamestart)
							));
				}
				gs.gamesCv->notify_one();
				gamestart = lineno + 1;
				gameId++;
			}
		}
	}	
	{
		std::lock_guard<std::mutex> lock(*gs.gamesMtx);
		for (int i=0; i<nMoveProcessors; i++) {
			gs.gamesQ->push(std::make_shared<GameData>(gs.pid, "FILE_DONE", gameId));
		}
	}
	gs.gamesCv->notify_all();
}


std::vector<std::shared_ptr<std::thread> > startGamesReader(
		std::queue<std::shared_ptr<GameData> >& gamesQ, 
		std::mutex& gamesMtx,
	   	std::condition_variable& gamesCv,
		std::string zst,
		std::vector<size_t>& frameBoundaries,
	   	int nMoveProcessors, int minSec, int maxSec, int maxInc) {
	GameState gs(&gamesQ, &gamesMtx, &gamesCv);
	std::vector<std::shared_ptr<std::thread> > procs;
	for (int i=0; i<frameBoundaries.size()-1; i++) {
		size_t start = frameBoundaries[i];
		size_t end = frameBoundaries[i+1];
		gs.pid = i;
		procs.push_back(
				std::make_shared<std::thread>(
					loadGamesZst, gs, zst, start, end, nMoveProcessors, minSec, maxSec, maxInc
					)
				);
	}
	return procs;
}

struct ProcessorState {
	std::queue<std::shared_ptr<GameData> >* gamesQ;
	std::queue<std::shared_ptr<MoveData> >* outputQ;
	int pid;
	std::mutex* gamesMtx;
	std::mutex* outputMtx;
	std::condition_variable* gamesCv;
	std::condition_variable* outputCv;
	ProcessorState(
			std::queue<std::shared_ptr<GameData> >* gamesQ, 
			std::queue<std::shared_ptr<MoveData> >* outputQ, 
			std::mutex* gamesMtx, 
			std::mutex* outputMtx,
		   	std::condition_variable* gamesCv,
		   	std::condition_variable* outputCv) 
		: gamesQ(gamesQ), 
		outputQ(outputQ),
	   	pid(-1),
	   	gamesMtx(gamesMtx),
	   	outputMtx(outputMtx),
	   	gamesCv(gamesCv),
	   	outputCv(outputCv) {};
};

void processGames(ProcessorState ps, int nReaders) {		
	int nReadersDone = 0;
	std::unordered_set<int> readerPids;
	int totalGames = 0;
	while(true) {
		std::shared_ptr<GameData> gd;
		{
			std::unique_lock<std::mutex> lock(*ps.gamesMtx);
			ps.gamesCv->wait(lock, [&]{return !ps.gamesQ->empty();});
			gd = ps.gamesQ->front();
			if (gd->info == "FILE_DONE") {
				if (!readerPids.contains(gd->pid)) {
					readerPids.insert(gd->pid);
					nReadersDone++;
					totalGames += gd->gameId;
					ps.gamesQ->pop();
				}
			} else {
				ps.gamesQ->pop();
			}
			lock.unlock();
		}
		if (gd->info == "FILE_DONE") {
			if (nReadersDone==nReaders) {
				std::lock_guard<std::mutex> lock(*ps.outputMtx);
				ps.outputQ->push(std::make_shared<MoveData>(ps.pid, "DONE", totalGames));
				ps.outputCv->notify_one();
				break;
			}
		} else {
			try {
				auto [mvs, clk, result] = parseMoves(gd->moveStr);
				{
					std::lock_guard<std::mutex> lock(*ps.outputMtx);
					ps.outputQ->push(
						std::make_shared<MoveData>(gd->pid, gd, mvs, clk, result)
					);
					ps.outputCv->notify_one();
				}
			} catch(std::exception &e) {
				std::lock_guard<std::mutex> lock(*ps.outputMtx);
				ps.outputQ->push(std::make_shared<MoveData>(gd->pid, "INVALID"));
			}
		}	
	}
}

std::vector<std::shared_ptr<std::thread> >  startProcessorThreads(
		int nMoveProcessors,
		int nReaders, 
		std::queue<std::shared_ptr<GameData> >& gamesQ, 
		std::queue<std::shared_ptr<MoveData> >& outputQ, 
		std::mutex& gamesMtx, 
		std::mutex& outputMtx,
	   	std::condition_variable& gamesCv, 
		std::condition_variable& outputCv) {

	std::vector<std::shared_ptr<std::thread> > threads;
	ProcessorState ps(&gamesQ, &outputQ, &gamesMtx, &outputMtx, &gamesCv, &outputCv);
	for (int i=0; i<nMoveProcessors; i++) {
		ps.pid = i;
		threads.push_back(std::make_shared<std::thread>(processGames, ps, nReaders));
	}	
	return threads;
}

ParallelParser::ParallelParser(int nReaders, int nMoveProcessors, int minSec, int maxSec, int maxInc) 
	: nReaders(nReaders), nMoveProcessors(nMoveProcessors), minSec(minSec), maxSec(maxSec), maxInc(maxInc) {};

ParallelParser::~ParallelParser() {
	for (auto gt: gameThreads) {
		gt->join();
	}
	for (auto rt: procThreads) {
		rt->join();
	}
}

std::shared_ptr<ParserOutput> ParallelParser::parse(std::string zst, std::string name, int printFreq) {
	auto output = std::make_shared<ParserOutput>();
	
	int64_t ngames = 0;
	int64_t nGamesLastUpdate = 0;
	int totalGames = INT_MAX;
	int nFinished = 0;
	
	std::vector<size_t> frameBoundaries = getFrameBoundaries(zst, nReaders);
	nReaders = frameBoundaries.size()-1;

	procThreads = startProcessorThreads(
		nMoveProcessors,
		nReaders, 
		gamesQ, 
		outputQ, 
		gamesMtx,
		outputMtx,
		gamesCv,
		outputCv
	);

	gameThreads = startGamesReader(
		gamesQ, 
		gamesMtx, 
		gamesCv, 
		zst,
		frameBoundaries,
		nMoveProcessors,
		minSec,
		maxSec,
		maxInc
	);

	auto start = hrc::now();
	auto lastPrintTime = start;
	while (ngames < totalGames || nFinished < nMoveProcessors) {
		std::shared_ptr<MoveData> md;
		{
			std::unique_lock<std::mutex> lock(outputMtx);
			outputCv.wait(lock, [&]{return !outputQ.empty();});
			md = outputQ.front();
			outputQ.pop();
			lock.unlock();
		}
		if (md->info == "DONE") {
			totalGames = md->gameId;
			nFinished++;
		} else if (md->info == "ERROR") {
			ngames++;
		} else if (md->info == "INVALID") {
			ngames++;
		} else if (md->info == "GAME") {
			output->welos.push_back(md->welo);
			output->belos.push_back(md->belo);
			output->timeCtl.push_back(md->time);
			output->increment.push_back(md->inc);
			output->mvs.push_back(md->mvs);
			output->clk.push_back(md->clk);
			output->result.push_back(md->result);
			ngames++;
			
			int totalGamesEst = ngames / md->progress;
			if (ellapsedGTE(lastPrintTime, printFreq)) {
				auto [eta, now] = getEta(totalGamesEst, ngames, start);
				long ellapsed = std::chrono::duration_cast<milli>(now-lastPrintTime).count();
				int gamesPerSec = 1000*(ngames-nGamesLastUpdate)/ellapsed;
				std::string status = name + ": parsed " + std::to_string(ngames) + \
									 " games (pid " + std::to_string(md->pid) + " " + std::to_string(int(100*md->progress)) + \
									 "% done, games/sec: " + \
									 std::to_string(gamesPerSec) + \
									 ", eta: " + eta + ")";
				std::cout << status << std::endl;

				nGamesLastUpdate = ngames;
				lastPrintTime = now;
			}
		} else {
			throw std::runtime_error("invalid code: " + md->info);
		}
	}
	return output;
}
