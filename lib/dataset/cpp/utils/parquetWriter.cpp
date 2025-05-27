#include "parquetWriter.h"

ParquetWriter::ParquetWriter(std::string root_path): root_path(root_path) {
    schema =
		arrow::schema({arrow::field("moves", arrow::utf8()), 
            arrow::field("clk", arrow::utf8()), 
            arrow::field("eval", arrow::utf8()), 
            arrow::field("result", arrow::int8()),
            arrow::field("welo", arrow::int16()),
            arrow::field("belo", arrow::int16()),
            arrow::field("timeCtl", arrow::int16()), 
            arrow::field("increment", arrow::int16())});

    PARQUET_ASSIGN_OR_THROW(outfile, arrow::io::FileOutputStream::Open(root_path + "/data.parquet"));

    parquet::WriterProperties::Builder builder;
    builder.compression(parquet::Compression::ZSTD);
    std::shared_ptr<parquet::WriterProperties> props = builder.build();

    auto pool = arrow::default_memory_pool();
    PARQUET_ASSIGN_OR_THROW(parquet_writer, parquet::arrow::FileWriter::Open(*schema, pool, outfile, props));

    mv_builder = arrow::StringBuilder(pool); 	
    clk_builder = arrow::StringBuilder(pool);	
    eval_builder = arrow::StringBuilder(pool);	
    welo_builder = arrow::NumericBuilder<arrow::Int16Type>(pool);	
    belo_builder = arrow::NumericBuilder<arrow::Int16Type>(pool);	
    timeCtl_builder = arrow::NumericBuilder<arrow::Int16Type>(pool);	
    increment_builder = arrow::NumericBuilder<arrow::Int16Type>(pool);	
    result_builder = arrow::NumericBuilder<arrow::Int8Type>(pool);
}

arrow::Result<std::string> ParquetWriter::write(std::shared_ptr<ParsedData> res) {
    mv_builder.Reset();
    clk_builder.Reset();
    welo_builder.Reset();
    belo_builder.Reset();
    timeCtl_builder.Reset();
    increment_builder.Reset();
    result_builder.Reset();
    eval_builder.Reset();

	size_t nRows = res->mvs.size();
    for (size_t j=0; j<nRows; j++) {
        ARROW_RETURN_NOT_OK(mv_builder.Append(res->mvs[j]));
        ARROW_RETURN_NOT_OK(clk_builder.Append(res->clk[j]));
        ARROW_RETURN_NOT_OK(eval_builder.Append(res->eval[j]));
        ARROW_RETURN_NOT_OK(welo_builder.Append(res->welos[j]));
        ARROW_RETURN_NOT_OK(belo_builder.Append(res->belos[j]));
        ARROW_RETURN_NOT_OK(timeCtl_builder.Append(res->timeCtl[j]));
        ARROW_RETURN_NOT_OK(increment_builder.Append(res->increment[j]));
        ARROW_RETURN_NOT_OK(result_builder.Append(res->result[j]));
    }

    std::shared_ptr<arrow::Array> moves;
    std::shared_ptr<arrow::Array> clk;
    std::shared_ptr<arrow::Array> welos;
    std::shared_ptr<arrow::Array> belos;
    std::shared_ptr<arrow::Array> timeCtl;
    std::shared_ptr<arrow::Array> increment;
    std::shared_ptr<arrow::Array> result;
    std::shared_ptr<arrow::Array> eval;

    ARROW_RETURN_NOT_OK(mv_builder.Finish(&moves));
    ARROW_RETURN_NOT_OK(clk_builder.Finish(&clk));
    ARROW_RETURN_NOT_OK(eval_builder.Finish(&eval));
    ARROW_RETURN_NOT_OK(welo_builder.Finish(&welos));
    ARROW_RETURN_NOT_OK(belo_builder.Finish(&belos));
    ARROW_RETURN_NOT_OK(timeCtl_builder.Finish(&timeCtl));
    ARROW_RETURN_NOT_OK(increment_builder.Finish(&increment));
    ARROW_RETURN_NOT_OK(result_builder.Finish(&result));

    auto batch = arrow::RecordBatch::Make(schema, nRows, {moves, clk, eval, result, welos, belos, timeCtl, increment});
    PARQUET_THROW_NOT_OK(parquet_writer->WriteRecordBatch(*batch));

    return root_path;
}

ParquetWriter::~ParquetWriter() {
    PARQUET_THROW_NOT_OK(parquet_writer->Close());
    PARQUET_THROW_NOT_OK(outfile->Close());
}