#include <string>
#include <iostream>
#include <chrono>
#include <filesystem>
#include "pgnProcessing/parserPool.h"
#include "utils/utils.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include <absl/strings/str_split.h>

ABSL_FLAG(std::string, zst, "", "comma-separated list of .zst archives to decompress and parse");
ABSL_FLAG(std::string, name, "", "comma-separated list of human-readable names for archives");
ABSL_FLAG(std::string, outdir, ".", "output directory to store parquet output files");
ABSL_FLAG(int, printFreq, 60, "Print status every printFreq seconds");
ABSL_FLAG(int, nSimultaneous, 1, "Number of zsts to process in parallel");
ABSL_FLAG(int, nReaders, 2, "Number of zst/pgn readers for parallel processing");
ABSL_FLAG(int, nMoveProcessors, std::thread::hardware_concurrency()-3, "Number of game parsers for parallel processing");
ABSL_FLAG(int, minSec, 300, "Minimum time control for game in seconds");
ABSL_FLAG(int, maxSec, 10800, "Maximum time control for game in seconds");
ABSL_FLAG(int, maxInc, 60, "Maximum increment for game in seconds");
ABSL_FLAG(size_t, chunkSize, 100000, "Number of games to process before writing to parquet");
ABSL_FLAG(std::string, elo_edges, "1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,4000",
          "Comma-separated list of Elo rating edges for organizing games");

int main(int argc, char *argv[]) {
	absl::SetProgramUsageMessage("Decompress and parse lichess .zst game archives into parquet files for use with training mimicChess network");
	absl::ParseCommandLine(argc, argv);

	std::string outdir = std::filesystem::absolute(absl::GetFlag(FLAGS_outdir)).string();
	
	ParserPool parserPool(
		absl::GetFlag(FLAGS_nReaders),
		absl::GetFlag(FLAGS_nMoveProcessors),
		absl::GetFlag(FLAGS_minSec),
		absl::GetFlag(FLAGS_maxSec),
		absl::GetFlag(FLAGS_maxInc),
		outdir,
		absl::GetFlag(FLAGS_elo_edges),
		absl::GetFlag(FLAGS_chunkSize),
		absl::GetFlag(FLAGS_printFreq),
		absl::GetFlag(FLAGS_nSimultaneous),
		1
		);

	std::vector<std::string> zsts = absl::StrSplit(absl::GetFlag(FLAGS_zst), ',');
	std::vector<std::string> names = absl::StrSplit(absl::GetFlag(FLAGS_name), ',');
	if (zsts.size() != names.size()) {
		std::cerr << "zst and name must have the same length\n";
		return 1;
	}
	for (size_t i = 0; i < zsts.size(); i++) {
		parserPool.enqueue(zsts[i], names[i]);
	}
	parserPool.join();
	return 0;
}
