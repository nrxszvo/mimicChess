#include "eloWriter.h"
#include <filesystem>
#include <absl/strings/str_split.h>
#include <iostream>

namespace fs = std::filesystem;

void processBatches(EloWriter* writer, std::queue<std::shared_ptr<ParsedData> >& batchQ, std::mutex& batchMtx, std::condition_variable& batchCv) {
    while (true) {
        std::shared_ptr<ParsedData> batch;
        {
            std::unique_lock<std::mutex> lock(batchMtx);
            batchCv.wait(lock, [&]{return !batchQ.empty();});
            batch = batchQ.front();
            batchQ.pop();
            lock.unlock();
        }
        if (batch->result.size() == 0) {
            break;
        }
        writer->writeBatch(batch);
    }
}

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

EloWriter::EloWriter(std::string output_dir, std::string elo_edges_str, int64_t chunk_size):
    elo_edges(ParseEloEdges(elo_edges_str)),
    chunk_size(chunk_size) {

    fs::path base_output_dir(output_dir);
    EnsureDirectoryExists(base_output_dir);

    // Create subdirectories
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

	procBatchThread = std::make_shared<std::thread>(processBatches, this, std::ref(batchQ), std::ref(batchMtx), std::ref(batchCv));
}

void EloWriter::close() {
    {
        std::lock_guard<std::mutex> lock(batchMtx);
        batchQ.push(std::make_shared<ParsedData>());
    }
    batchCv.notify_all();
    procBatchThread->join();

    for (int i = 0; i< elo_edges.size(); i++) {
        for (int j = 0; j < elo_edges.size(); j++) {
            if (data[i][j]->mvs.size() > 0) {
                auto result = writers[i][j]->write(data[i][j]);
                if (!result.ok()) {
                    throw std::runtime_error("Error writing table: " + result.status().ToString());
                }
            }
            writers[i][j]->close();
        }
    }
}

void EloWriter::queueBatch(std::shared_ptr<ParsedData> batch) {
    {
        std::lock_guard<std::mutex> lock(batchMtx);
        batchQ.push(batch);
    }
    batchCv.notify_one();
}


void EloWriter::writeBatch(std::shared_ptr<ParsedData> batch) {
    for (size_t i = 0; i < batch->result.size(); ++i) {
        int wElo = batch->welos[i];
        int bElo = batch->belos[i];
        size_t wBucket = GetEloBucket(wElo, elo_edges);
        size_t bBucket = GetEloBucket(bElo, elo_edges);
        auto ij_data = data[wBucket][bBucket];
        ij_data->mvs.push_back(batch->mvs[i]);
        ij_data->clk.push_back(batch->clk[i]);
        ij_data->eval.push_back(batch->eval[i]);
        ij_data->result.push_back(batch->result[i]);
        ij_data->welos.push_back(wElo);
        ij_data->belos.push_back(bElo);
        ij_data->timeCtl.push_back(batch->timeCtl[i]);
        ij_data->increment.push_back(batch->increment[i]);

        if (ij_data->mvs.size() >= chunk_size) {
            auto result = writers[wBucket][bBucket]->write(ij_data);
            if (!result.ok()) {
                throw std::runtime_error("Error writing table: " + result.status().ToString());
            }
            data[wBucket][bBucket] = std::make_shared<ParsedData>();
        }
    }
}
