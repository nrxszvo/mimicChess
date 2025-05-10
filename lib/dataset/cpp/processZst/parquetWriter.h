#ifndef PARQUETWRITER_H
#define PARQUETWRITER_H
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include "parser.h"

class ParquetWriter {
public:
    ParquetWriter(std::string root_path);
    ~ParquetWriter();
    arrow::Result<std::string> write(std::shared_ptr<ParserOutput> res);
private:
    std::string root_path;
    std::shared_ptr<arrow::Schema> schema;
    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    std::unique_ptr<parquet::arrow::FileWriter> parquet_writer;
    arrow::StringBuilder mv_builder;
    arrow::StringBuilder clk_builder;
    arrow::StringBuilder eval_builder;
    arrow::NumericBuilder<arrow::Int16Type> welo_builder;
    arrow::NumericBuilder<arrow::Int16Type> belo_builder;
    arrow::NumericBuilder<arrow::Int16Type> timeCtl_builder;
    arrow::NumericBuilder<arrow::Int16Type> increment_builder;
    arrow::NumericBuilder<arrow::Int8Type> result_builder;
};
#endif
