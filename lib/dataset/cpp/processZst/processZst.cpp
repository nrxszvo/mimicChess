#include <string>
#include <iostream>
#include <chrono>
#include <filesystem>
#include "npy.hpp"
#include "parallelParser.h"
#include "serialParser.h"
#include "lib/parseMoves.h"
#include "profiling/profiler.h"
#include "utils/utils.h"
#include "lib/decompress.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

ABSL_FLAG(std::string, zst, "", ".zst archive to decompress and parse");
ABSL_FLAG(std::string, name, "", "human-readable name for archive");
ABSL_FLAG(std::string, outdir, ".", "output directory to store npy output files");
ABSL_FLAG(bool, serial, false, "Disable parallel processing");
ABSL_FLAG(int, printFreq, 60, "Print status every printFreq seconds");
ABSL_FLAG(int, nReaders, 2, "Number of zst/pgn readers for parallel processing");
ABSL_FLAG(int, nMoveProcessors, std::thread::hardware_concurrency()-3, "Number of game parsers for parallel processing");
ABSL_FLAG(int, minSec, 300, "Minimum time control for game in seconds");
ABSL_FLAG(int, maxSec, 10800, "Maximum time control for game in seconds");
ABSL_FLAG(int, maxInc, 60, "Maximum increment for game in seconds");
ABSL_FLAG(size_t, chunkSize, 100000, "Number of rows to write per chunk");

arrow::Result<std::string> writeParquet(std::string& root_path, std::shared_ptr<ParserOutput> res, size_t chunk_size) {
    auto pool = arrow::default_memory_pool();
	auto schema =
		arrow::schema({arrow::field("moves", arrow::utf8()), arrow::field("clk", arrow::utf8()), arrow::field("result", arrow::int8()),
						arrow::field("welo", arrow::int16()),
						arrow::field("belo", arrow::int16()),
						arrow::field("timeCtl", arrow::int16()), arrow::field("increment", arrow::int16())});

    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    PARQUET_ASSIGN_OR_THROW(outfile, arrow::io::FileOutputStream::Open(root_path + "/data.parquet"));

    std::unique_ptr<parquet::arrow::FileWriter> parquet_writer;
    ARROW_ASSIGN_OR_RAISE(parquet_writer, parquet::arrow::FileWriter::Open(*schema, pool, outfile));

	arrow::StringBuilder mv_builder(pool); 	
    arrow::StringBuilder clk_builder(pool);	
    arrow::NumericBuilder<arrow::Int16Type> welo_builder(pool);	
    arrow::NumericBuilder<arrow::Int16Type> belo_builder(pool);	
    arrow::NumericBuilder<arrow::Int16Type> timeCtl_builder(pool);	
    arrow::NumericBuilder<arrow::Int16Type> increment_builder(pool);	
    arrow::NumericBuilder<arrow::Int8Type> result_builder(pool);

	size_t nRows = res->mvs.size();
    for (size_t i=0; i<nRows; i+=chunk_size) {

		auto batch_size = std::min(chunk_size, nRows-i);
		mv_builder.Reset();
		clk_builder.Reset();
		welo_builder.Reset();
		belo_builder.Reset();
		timeCtl_builder.Reset();
		increment_builder.Reset();
		result_builder.Reset();

		for (size_t j=0; j<batch_size; j++) {
			ARROW_RETURN_NOT_OK(mv_builder.Append(res->mvs[i+j]));
			ARROW_RETURN_NOT_OK(clk_builder.Append(res->clk[i+j]));
			ARROW_RETURN_NOT_OK(welo_builder.Append(res->welos[i+j]));
			ARROW_RETURN_NOT_OK(belo_builder.Append(res->belos[i+j]));
			ARROW_RETURN_NOT_OK(timeCtl_builder.Append(res->timeCtl[i+j]));
			ARROW_RETURN_NOT_OK(increment_builder.Append(res->increment[i+j]));
			ARROW_RETURN_NOT_OK(result_builder.Append(res->result[i+j]));
		}
        std::shared_ptr<arrow::Array> moves;
		std::shared_ptr<arrow::Array> clk;
        std::shared_ptr<arrow::Array> welos;
        std::shared_ptr<arrow::Array> belos;
        std::shared_ptr<arrow::Array> timeCtl;
        std::shared_ptr<arrow::Array> increment;
		std::shared_ptr<arrow::Array> result;

        ARROW_RETURN_NOT_OK(mv_builder.Finish(&moves));
        ARROW_RETURN_NOT_OK(clk_builder.Finish(&clk));
        ARROW_RETURN_NOT_OK(welo_builder.Finish(&welos));
        ARROW_RETURN_NOT_OK(belo_builder.Finish(&belos));
        ARROW_RETURN_NOT_OK(timeCtl_builder.Finish(&timeCtl));
        ARROW_RETURN_NOT_OK(increment_builder.Finish(&increment));
		ARROW_RETURN_NOT_OK(result_builder.Finish(&result));

        auto batch = arrow::RecordBatch::Make(schema, batch_size, {moves, clk, result, welos, belos, timeCtl, increment});
        PARQUET_THROW_NOT_OK(parquet_writer->WriteRecordBatch(*batch));
    }

    PARQUET_THROW_NOT_OK(parquet_writer->Close());
    PARQUET_THROW_NOT_OK(outfile->Close());

    return root_path;
}


void writeNpy(std::string outdir, std::shared_ptr<ParserOutput> res) {

	std::ofstream output_file(outdir + "/moves.data");
    std::ostream_iterator<std::string> output_iterator(output_file, "\n");
    std::copy(res->mvs.begin(), res->mvs.end(), output_iterator);

	npy::npy_data_ptr<int16_t> welo_ptr;
	npy::npy_data_ptr<int16_t> belo_ptr;
	npy::npy_data_ptr<int64_t> gs_ptr; 
	npy::npy_data_ptr<int16_t> timeCtl_ptr;
	npy::npy_data_ptr<int16_t> inc_ptr;
	
	welo_ptr.data_ptr = res->welos.data(); 	
	welo_ptr.shape = { res->welos.size() };
	belo_ptr.data_ptr = res->belos.data();
	belo_ptr.shape = { res->belos.size() };
	gs_ptr.data_ptr = res->gamestarts.data();
	gs_ptr.shape = { res->gamestarts.size() };
	timeCtl_ptr.data_ptr = res->timeCtl.data();
	timeCtl_ptr.shape = { res->timeCtl.size() };
	inc_ptr.data_ptr = res->increment.data();
	inc_ptr.shape = { res->increment.size() };

	npy::write_npy(outdir + "/welos.npy", welo_ptr);
	npy::write_npy(outdir + "/belos.npy", belo_ptr);
	npy::write_npy(outdir + "/gamestarts.npy", gs_ptr);
	npy::write_npy(outdir + "/timeCtl.npy", timeCtl_ptr);
	npy::write_npy(outdir + "/inc.npy", inc_ptr);
}

int main(int argc, char *argv[]) {
	absl::SetProgramUsageMessage("Decompress and parse lichess .zst game archives into npy files for use with training mimicChess network");
	absl::ParseCommandLine(argc, argv);
	auto start = std::chrono::high_resolution_clock::now();
	std::shared_ptr<ParserOutput> res;
	std::string name = absl::GetFlag(FLAGS_name);



	if (absl::GetFlag(FLAGS_serial)) {
		res = processSerial(absl::GetFlag(FLAGS_zst));
	} else {
		ParallelParser parser(
				absl::GetFlag(FLAGS_nReaders),
			   	absl::GetFlag(FLAGS_nMoveProcessors),
				absl::GetFlag(FLAGS_minSec),
				absl::GetFlag(FLAGS_maxSec),
				absl::GetFlag(FLAGS_maxInc)
				);
		res = parser.parse(absl::GetFlag(FLAGS_zst), 
				name,
				absl::GetFlag(FLAGS_printFreq)
				);
	}
	std::string outdir = std::filesystem::absolute(absl::GetFlag(FLAGS_outdir)).string();
	std::filesystem::create_directories(outdir);
	auto result = writeParquet(outdir, res, absl::GetFlag(FLAGS_chunkSize));
	if (!result.ok()) {
		std::cerr << "Error writing table: " << result.status() << std::endl;
		return 1;
	}
	auto stop = std::chrono::high_resolution_clock::now();
	std::cout << name << " finished parsing in " << getEllapsedStr(start, stop) << std::endl;
	profiler.report();
	return 0;
}
