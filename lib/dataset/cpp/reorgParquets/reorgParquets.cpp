#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/flags/usage.h>
#include <absl/strings/str_split.h>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>
#include "utils/parquetWriter.h"
#include "utils/parquetReader.h"
#include "parser.h"

namespace fs = std::filesystem;

// Define command line flags
ABSL_FLAG(std::string, input_dir, "", "Directory containing parquet files");
ABSL_FLAG(std::string, output_dir, "", "Output directory for reorganized files");
ABSL_FLAG(std::string, elo_edges, "1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,4000",
          "Comma-separated list of Elo rating edges for organizing games");
ABSL_FLAG(int64_t, row_group_size, 1024, "Number of rows per row group");

// Helper function to parse Elo edges
std::vector<int> ParseEloEdges(const std::string& elo_edges_str) {
    std::vector<std::string> edges = absl::StrSplit(elo_edges_str, ',');
    std::vector<int> result;
    result.reserve(edges.size());
    
    for (const auto& edge : edges) {
        result.push_back(std::stoi(edge));
    }
    std::sort(result.begin(), result.end());
    return result;
}

// Helper function to get Elo bucket for a rating
size_t GetEloBucket(int rating, const std::vector<int>& edges) {
    return std::upper_bound(edges.begin(), edges.end(), rating) - edges.begin();
}

// Helper function to ensure directory exists
void EnsureDirectoryExists(const fs::path& dir) {
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
}

int main(int argc, char* argv[]) {
    absl::SetProgramUsageMessage("Reorganize parquet files into directories based on Elo rating");
    absl::ParseCommandLine(argc, argv);

    std::string input_dir = absl::GetFlag(FLAGS_input_dir);
    std::string output_dir = absl::GetFlag(FLAGS_output_dir);
    auto elo_edges = ParseEloEdges(absl::GetFlag(FLAGS_elo_edges));
    int64_t rowGroupSize = absl::GetFlag(FLAGS_row_group_size);

    if (input_dir.empty() || output_dir.empty()) {
        std::cerr << "Error: Both input and output directories must be specified.\n";
        return 1;
    }

    // Create output directory structure
    fs::path base_output_dir(output_dir);
    EnsureDirectoryExists(base_output_dir);

    // Create subdirectories for each Elo range
    std::vector<std::vector<std::shared_ptr<ParquetWriter>>> writers;
    std::vector<std::vector<std::shared_ptr<ParsedData>>> data;
    for (size_t i = 0; i < elo_edges.size(); ++i) {
        std::vector<std::shared_ptr<ParquetWriter>> i_writers;
        std::vector<std::shared_ptr<ParsedData>> i_data;
        std::string welo = std::to_string(elo_edges[i]);
        for (size_t j = 0; j < elo_edges.size(); ++j) {
            std::string belo = std::to_string(elo_edges[j]);
            fs::path elo_dir = base_output_dir / welo / belo;
            EnsureDirectoryExists(elo_dir);
            i_writers.push_back(std::make_shared<ParquetWriter>(elo_dir.string()));
            i_data.push_back(std::make_shared<ParsedData>());
        }
        writers.push_back(i_writers);
        data.push_back(i_data);
    }

    // Process each parquet file in the input directory
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.path().extension() != ".parquet") continue;

        try {
            ParquetReader reader;
            arrow::Status status = reader.Open(entry.path().string());
            if (!status.ok()) {
                std::cerr << "Error opening file " << entry.path() << ": " << status.ToString() << std::endl;
                continue;
            }
            
            int64_t totalGames = reader.numRows();
            int64_t gamesProcessed = 0;
            int prevProgress = 0;
            
            // Process the file batch by batch
            ParsedData batch;
            while (reader.ReadBatch(batch).ok()) {
                if (batch.result.size() == 0) {
                    break;
                }
                for (size_t i = 0; i < batch.result.size(); ++i) {
                    int wElo = batch.welos[i];
                    int bElo = batch.belos[i];
                    size_t wBucket = GetEloBucket(wElo, elo_edges);
                    size_t bBucket = GetEloBucket(bElo, elo_edges);
                    auto ij_data = data[wBucket][bBucket];
                    ij_data->mvs.push_back(batch.mvs[i]);
                    ij_data->clk.push_back(batch.clk[i]);
                    ij_data->eval.push_back(batch.eval[i]);
                    ij_data->result.push_back(batch.result[i]);
                    ij_data->welos.push_back(wElo);
                    ij_data->belos.push_back(bElo);
                    ij_data->timeCtl.push_back(batch.timeCtl[i]);
                    ij_data->increment.push_back(batch.increment[i]);

                    if (ij_data->mvs.size() >= rowGroupSize) {
                        auto result = writers[wBucket][bBucket]->write(ij_data);
                        if (!result.ok()) {
                            throw std::runtime_error("Error writing table: " + result.status().ToString());
                        }
                        data[wBucket][bBucket] = std::make_shared<ParsedData>();
                    } 
                }
                gamesProcessed += batch.result.size();
                int progress = int((100.0 * gamesProcessed) / totalGames);
                if (progress > prevProgress) {
                    std::cout << entry.path().parent_path().filename() / entry.path().filename() << ": " << gamesProcessed << " games (" << progress << "% done)\r";
                    std::flush(std::cout);
                    prevProgress = progress;
                }
            }
            std::cout << "Finished " << entry.path().parent_path().filename() / entry.path().filename() << std::setw(50) << " " << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error processing " << entry.path() << ": " << e.what() << std::endl;
            continue;
        }
    }

    return 0;
}