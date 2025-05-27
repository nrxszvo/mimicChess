#include "parquetReader.h"
#include <arrow/result.h>
#include <arrow/table.h>
#include <arrow/io/api.h>
#include <iostream>

arrow::Status ParquetReader::Open(const std::string& file_path) {
    auto arrow_reader_props = parquet::ArrowReaderProperties();
    arrow_reader_props.set_batch_size(128 * 1024);  // default 64 * 1024

    parquet::arrow::FileReaderBuilder reader_builder;
    ARROW_RETURN_NOT_OK(reader_builder.OpenFile(file_path));
    reader_builder.memory_pool(arrow::default_memory_pool());
    reader_builder.properties(arrow_reader_props);

    ARROW_ASSIGN_OR_RAISE(parquet_reader, reader_builder.Build());
    ARROW_RETURN_NOT_OK(parquet_reader->GetSchema(&schema));
    ARROW_ASSIGN_OR_RAISE(batch_reader, parquet_reader->GetRecordBatchReader());
    return arrow::Status::OK();
}

int64_t ParquetReader::numRows() {
    int64_t num_rows = 0;
    if (parquet_reader) {   
        arrow::Status st = parquet_reader->ScanContents({2}, 1024, &num_rows);
        if (!st.ok()) {
            throw std::runtime_error(st.ToString());
        }
    }
    return num_rows;
}

arrow::Status ParquetReader::ReadBatch(ParsedData& output) {

    // Clear previous batch data
    output.mvs.clear();
    output.clk.clear();
    output.eval.clear();
    output.result.clear();
    output.welos.clear();
    output.belos.clear();
    output.timeCtl.clear();
    output.increment.clear();

    std::shared_ptr<arrow::RecordBatch> batch;
    ARROW_RETURN_NOT_OK(batch_reader->ReadNext(&batch));
    if (!batch) return arrow::Status::OK(); //EOF

    // Reserve space for the new batch
    output.mvs.reserve(batch->num_rows());
    output.clk.reserve(batch->num_rows());
    output.eval.reserve(batch->num_rows());
    output.result.reserve(batch->num_rows());
    output.welos.reserve(batch->num_rows());
    output.belos.reserve(batch->num_rows());
    output.timeCtl.reserve(batch->num_rows());
    output.increment.reserve(batch->num_rows());

    // Get arrays from batch
    auto moves_array = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName(schema->field(0)->name()));
    auto clk_array = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName(schema->field(1)->name()));
    auto eval_array = std::static_pointer_cast<arrow::StringArray>(batch->GetColumnByName(schema->field(2)->name()));
    auto result_array = std::static_pointer_cast<arrow::Int8Array>(batch->GetColumnByName(schema->field(3)->name()));
    auto welo_array = std::static_pointer_cast<arrow::Int16Array>(batch->GetColumnByName(schema->field(4)->name()));
    auto belo_array = std::static_pointer_cast<arrow::Int16Array>(batch->GetColumnByName(schema->field(5)->name()));
    auto timeCtl_array = std::static_pointer_cast<arrow::Int16Array>(batch->GetColumnByName(schema->field(6)->name()));
    auto increment_array = std::static_pointer_cast<arrow::Int16Array>(batch->GetColumnByName(schema->field(7)->name()));

    // Copy data from the batch
    for (int64_t i = 0; i < batch->num_rows(); i++) {
        output.mvs.push_back(moves_array->GetString(i));
        output.clk.push_back(clk_array->GetString(i));
        output.eval.push_back(eval_array->GetString(i));
        output.result.push_back(result_array->Value(i));
        output.welos.push_back(welo_array->Value(i));
        output.belos.push_back(belo_array->Value(i));
        output.timeCtl.push_back(timeCtl_array->Value(i));
        output.increment.push_back(increment_array->Value(i));
    }

    return arrow::Status::OK();
}