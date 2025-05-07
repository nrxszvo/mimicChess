#include "lib/parseMoves.h"
#include "lib/decompress.h"
#include "lib/validate.h"
#include "parser.h"
#include "utils/utils.h"
#include "profiling/profiler.h"
#include "serialParser.h"
#include <filesystem>

std::shared_ptr<ParserOutput> processSerial(std::string zst) {

	std::vector<size_t> frameBoundaries = getFrameBoundaries(zst, 1);
	DecompressStream decompressor(zst,0,frameBoundaries[1]);
	PgnProcessor processor(300, 10800, 60);

	int gamestart = 0;
	int lineno = 0;

	auto output = std::make_shared<ParserOutput>();
	
	uintmax_t nbytes = std::filesystem::file_size(zst);	
	size_t bytesRead;
	uintmax_t bytesProcessed = 0;

	int progress = 0;
	int printFreq = 10;

	profiler.init("parseMoves");
	profiler.init("regex");
	profiler.init("decompress");

	auto start = hrc::now();
	while((bytesRead = decompressor.decompressFrame()) != 0) {
		std::vector<std::string> lines;
		decompressor.getLines(lines);
		bytesProcessed += bytesRead;
		for (auto line: lines) {
			lineno++;
			std::string code = processor.processLine(line);		
			if (code == "COMPLETE") {
				profiler.start("parseMoves");
				auto [moves, clk, eval, result] = parseMoves(processor.getMoveStr());
				profiler.stop("parseMoves");
				output->welos.push_back(processor.getWelo());
				output->belos.push_back(processor.getBelo());
				output->timeCtl.push_back(processor.getTime());
				output->increment.push_back(processor.getInc());
				output->mvs.push_back(moves);
				output->clk.push_back(clk);
				output->eval.push_back(eval);
				output->result.push_back(result);
				int ngames = output->gamestarts.size();
				int totalGamesEst = ngames / ((double)bytesProcessed / nbytes);
				int curProg = int((100.0 / printFreq) * ngames / totalGamesEst);
				if (curProg > progress) {
					progress = curProg;
					auto [eta, gamesPerSec] = getEta(totalGamesEst, ngames, start);
					std::string status = "parsed " + std::to_string(ngames) + \
										 " games (" + std::to_string(printFreq*progress) + \
										 "% done, eta: " + eta + ")";
					std::cout << status << '\r' << std::flush;
				}
				gamestart = lineno+1;
			}
		}
	}
	return output;	
}
