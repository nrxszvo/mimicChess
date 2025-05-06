#ifndef MC_PARSER_H
#define MC_PARSER_H
#include <memory>
#include <vector>
#include <string>

struct ParserOutput {
	std::vector<int16_t> welos;
	std::vector<int16_t> belos;
	std::vector<int16_t> timeCtl;
	std::vector<int16_t> increment;
	std::vector<int64_t> gamestarts;
	std::vector<std::string> mvs;
	std::vector<int8_t> result;
	std::vector<std::string> clk;
};
#endif
