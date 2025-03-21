#ifndef MC_PARSER_H
#define MC_PARSER_H
#include <memory>
#include <vector>

struct ParserOutput {
	std::vector<int16_t> welos;
	std::vector<int16_t> belos;
	std::vector<int16_t> timeCtl;
	std::vector<int16_t> increment;
	std::vector<int64_t> gamestarts;
	std::vector<int16_t> mvids;
	std::vector<int16_t> clk;
};
#endif
