#ifndef PARQUETREADER_H
#define PARQUETREADER_H

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include "include/parser.h"


class ParquetReader {
public:
    arrow::Status Open(const std::string& file_path);
    arrow::Status ReadBatch(ParsedData& batch);
    int64_t numRows();

private:
    std::string file_path;
    std::shared_ptr<arrow::io::ReadableFile> infile;
    std::unique_ptr<parquet::arrow::FileReader> parquet_reader;
    std::shared_ptr<arrow::Schema> schema;
    std::shared_ptr<arrow::RecordBatchReader> batch_reader;
};

#endif
