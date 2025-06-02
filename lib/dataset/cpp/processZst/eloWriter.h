#ifndef ELO_WRITER_H
#define ELO_WRITER_H

#include "parser.h"
#include "utils/parquetWriter.h"
#include <queue>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>

class EloWriter {
public:
    EloWriter(std::string output_dir, std::string elo_edges_str, int64_t chunk_size);
    void close();
    void queueBatch(std::shared_ptr<ParsedData> batch);
    void writeBatch(std::shared_ptr<ParsedData> batch);
private:
	std::queue<std::shared_ptr<ParsedData> > batchQ;
    std::mutex batchMtx;
    std::condition_variable batchCv;
    std::shared_ptr<std::thread> procBatchThread;
    std::vector<std::vector<std::shared_ptr<ParquetWriter>>> writers;
    std::vector<std::vector<std::shared_ptr<ParsedData>>> data;
    std::vector<int> elo_edges;
    int64_t chunk_size;
};

#endif
