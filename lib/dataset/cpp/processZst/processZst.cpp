#include <string>
#include <iostream>
#include <chrono>
#include <filesystem>
#include "parallelParser.h"
#include "serialParser.h"
#include "lib/parseMoves.h"
#include "profiling/profiler.h"
#include "utils/utils.h"
#include "lib/decompress.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "utils/parquetWriter.h"

ABSL_FLAG(std::string, zst, "", ".zst archive to decompress and parse");
ABSL_FLAG(std::string, name, "", "human-readable name for archive");
ABSL_FLAG(std::string, outdir, ".", "output directory to store parquet output files");
ABSL_FLAG(bool, serial, false, "Disable parallel processing");
ABSL_FLAG(int, printFreq, 60, "Print status every printFreq seconds");
ABSL_FLAG(int, nReaders, 2, "Number of zst/pgn readers for parallel processing");
ABSL_FLAG(int, nMoveProcessors, std::thread::hardware_concurrency()-3, "Number of game parsers for parallel processing");
ABSL_FLAG(int, minSec, 300, "Minimum time control for game in seconds");
ABSL_FLAG(int, maxSec, 10800, "Maximum time control for game in seconds");
ABSL_FLAG(int, maxInc, 60, "Maximum increment for game in seconds");
ABSL_FLAG(size_t, rowGroupSize, 100000, "Parquet row group size");
ABSL_FLAG(int, procId, 0, "Process ID");

int main(int argc, char *argv[]) {
	absl::SetProgramUsageMessage("Decompress and parse lichess .zst game archives into parquet files for use with training mimicChess network");
	absl::ParseCommandLine(argc, argv);

	auto start = std::chrono::high_resolution_clock::now();
	std::string name = absl::GetFlag(FLAGS_name);
	std::string outdir = std::filesystem::absolute(absl::GetFlag(FLAGS_outdir)).string();
	std::filesystem::create_directories(outdir);
	int64_t ngames;
	int procId = absl::GetFlag(FLAGS_procId);

	if (absl::GetFlag(FLAGS_serial)) {
		auto res = processSerial(absl::GetFlag(FLAGS_zst));
		auto writer = ParquetWriter(outdir);
		auto result = writer.write(res);
		if (!result.ok()) {
			std::cerr << "Error writing table: " << result.status() << std::endl;
			return 1;
		}
		ngames = res->mvs.size();
	} else {
		ParallelParser parser(
			absl::GetFlag(FLAGS_nReaders),
			absl::GetFlag(FLAGS_nMoveProcessors),
			absl::GetFlag(FLAGS_minSec),
			absl::GetFlag(FLAGS_maxSec),
			absl::GetFlag(FLAGS_maxInc),
			outdir,
			absl::GetFlag(FLAGS_rowGroupSize)
			);
		ngames = parser.parse(absl::GetFlag(FLAGS_zst), 
			name,
			procId,
			absl::GetFlag(FLAGS_printFreq)
			);
	}

	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << "\033[" << procId << "H\033[Kzst proc: " << name << " finished parsing " << ngames << " games in " << getEllapsedStr(start, stop) << "\r";
	for (int i = 0; i < absl::GetFlag(FLAGS_nReaders); i++) {
		std::cout << "\033[" << procId + i + 1 << "H\033[K\r";
	}
	profiler.report();
	return 0;
}
