#include <vector>
#include <string>
#include <fstream>
#include <memory>

struct GameMD {
	int64_t gameIdx;
	int64_t gameStart;
	int16_t timeCtl;
	int16_t inc;
	int16_t welo;
	int16_t belo;
};

class MMCRawDataReader {		
	std::ifstream gsIf;
	std::ifstream tcIf;
	std::ifstream incIf;
	std::ifstream clkIf;
	std::ifstream weloIf;
	std::ifstream beloIf;
	std::string npydir;
	int64_t totalGames;
	int64_t offsetStartGame;
	int64_t nGames;
	int64_t gameStart;
public:
	MMCRawDataReader(std::string npydir, int64_t startGame=0, int64_t nGamesToProcess =-1);
	std::shared_ptr<GameMD> nextGame(std::vector<int16_t>& clk);
	int64_t getTotalGames();
};
