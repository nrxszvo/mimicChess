#ifndef PARSER_POOL_H
#define PARSER_POOL_H
#include <condition_variable>
#include <functional>
#include <chrono>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include "parallelParser.h"
#include "eloWriter.h"
#include "utils/utils.h"

class ParserPool {
public:
    ParserPool(int nReaders, int nMoveProcessors, int minSec, int maxSec, int maxInc, std::string outdir, std::string elo_edges, size_t chunkSize, int printFreq, size_t numThreads, int printOffset)
        : stop_(false), curProcess(0), nCompleted(0)
    {
		writer = std::make_shared<EloWriter>(outdir, elo_edges, chunkSize);
        std::cout << "\033[2J" << std::flush;
        threads_.reserve(numThreads);
        for (size_t procId = 0; procId < numThreads; ++procId) {
            threads_.emplace_back([=, this] {
                int thisProc;
                ParallelParser parser(nReaders, nMoveProcessors, minSec, maxSec, maxInc, writer, chunkSize);
                while (true) {
                    std::string zst;
                    std::string name;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        cv_.wait(lock, [this] {
                            return !tasks_.empty() || stop_;
                        });
                        if (stop_ && tasks_.empty()) {
                            return;
                        }
                        std::tie(zst, name) = tasks_.front();
                        tasks_.pop();
                        thisProc = curProcess++;
                    }
                    auto start = std::chrono::high_resolution_clock::now();
                    auto offset = printOffset + procId * (2+nReaders);
                    {
                        std::lock_guard<std::mutex> lock(print_mutex_);
                        std::cout << "\033[" << offset << "H\033[K" << thisProc << ": parsing " << name << "..." << std::flush;
                    }
                    auto ngames = parser.parse(zst, name, offset, printFreq, print_mutex_);
                    auto stop = std::chrono::high_resolution_clock::now();
                    {
                        std::lock_guard<std::mutex> lock(print_mutex_);
                        std::cout << "\033[" << offset << "H\033[K" << thisProc << ": finished parsing " << ngames << " games from " << name << " in " << getEllapsedStr(start, stop);
                        for (int i = 1; i <= nReaders; i++) {
                            std::cout << "\033[" << i + offset << "H\033[K";
                        }
                        std::cout << "\033[" << printOffset + numThreads * (2+nReaders) << "H" << std::flush;
                        nCompleted++;
                    }
                }
            });
        }
    }

    void join() {
        stop_ = true;
        cv_.notify_all();
        for (auto& thread : threads_) {
            thread.join();
        }
        writer->close();
    }

    void enqueue(std::string zst, std::string name)
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            tasks_.emplace(zst, name);
        }
        cv_.notify_one();
    }

    int numCompleted() {
        return nCompleted;
    }

private:
    std::shared_ptr<EloWriter> writer;
    std::vector<std::thread> threads_;
    std::queue<std::pair<std::string, std::string> > tasks_;
    std::mutex queue_mutex_;
    std::mutex print_mutex_;
    std::condition_variable cv_;
    bool stop_;
    int curProcess;
    int nCompleted;
};

#endif
