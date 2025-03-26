#include "MMCRawDataReader.h"
#include <re2/re2.h>
#include <stdexcept>

MMCRawDataReader::MMCRawDataReader(std::string npydir, int64_t startGame, int64_t nGamesToProcess): npydir(npydir), offsetStartGame(startGame), nGames(0) {
	gsIf = std::ifstream(npydir + "/gamestarts.npy", std::ios::binary);
	tcIf = std::ifstream(npydir + "/timeCtl.npy", std::ios::binary); 
	incIf = std::ifstream(npydir + "/inc.npy", std::ios::binary);
	weloIf = std::ifstream(npydir + "/welos.npy", std::ios::binary);
	beloIf = std::ifstream(npydir + "/belos.npy", std::ios::binary);

	int64_t startByte64 = offsetStartGame*sizeof(int64_t);
	int64_t startByte16 = offsetStartGame*sizeof(int16_t);

	tcIf.seekg(startByte16, tcIf.beg);
	incIf.seekg(startByte16, incIf.beg);
	weloIf.seekg(startByte16, weloIf.beg);
	beloIf.seekg(startByte16, beloIf.beg);

	if (nGamesToProcess == -1) {
		gsIf.seekg(0, gsIf.end);
		size_t nbytes = gsIf.tellg()-startByte64;
		totalGames = nbytes/sizeof(int64_t);
	} else {
		totalGames = nGamesToProcess;
	}
	
	gsIf.seekg(startByte64, gsIf.beg);
	gsIf.read((char*)&gameStart, sizeof(gameStart));

	clkIf = std::ifstream(npydir + "/clk.npy", std::ios::binary);
	clkIf.seekg(gameStart*sizeof(int16_t), clkIf.beg);
}

int64_t MMCRawDataReader::getTotalGames() {
	return totalGames;
}

std::shared_ptr<GameMD> MMCRawDataReader::nextGame(std::vector<int16_t>& clkVec) {
	
	clkVec.clear();

	if (nGames == totalGames) return nullptr;

	int64_t gameEnd;	
	int64_t gameSize;

	while (true) {
		gsIf.read((char*)&gameEnd, sizeof(gameEnd));
		if (gsIf.gcount() <= 0) {
			int64_t pos = clkIf.tellg();
			clkIf.seekg(0, clkIf.end);
			int64_t rem = clkIf.tellg() - pos;
			clkIf.seekg(pos, clkIf.beg);
			gameSize = rem/sizeof(int16_t);
			if (nGames != totalGames-1) throw std::runtime_error("permaturely reached eof (nGames=" + std::to_string(nGames) + ", total games=" + std::to_string(totalGames) + ") in " + npydir);
		} else if (gameEnd == 0) {
			throw std::runtime_error("gameEnd is 0");
		} else {
			gameSize = gameEnd-gameStart;
		}
		if (gameSize > 0) break;
		totalGames--;
	}

	size_t nbytes = gameSize*sizeof(int16_t);
	int16_t* buf = (int16_t*)malloc(nbytes);
	clkIf.read((char*)buf, nbytes);
	clkVec.reserve(gameSize);
	clkVec.insert(clkVec.begin(), buf, &buf[gameSize]);
	free(buf);

	int64_t gs = gameStart;
	int16_t tc;
	int16_t inc;
	int16_t we;	
	int16_t be;
	int64_t gIdx = offsetStartGame + nGames;
	
	gameStart = gameEnd;
	nGames++;

	tcIf.read(reinterpret_cast<char*>(&tc), sizeof(tc));
	incIf.read(reinterpret_cast<char*>(&inc), sizeof(inc));
	weloIf.read(reinterpret_cast<char*>(&we), sizeof(we));
	beloIf.read(reinterpret_cast<char*>(&be), sizeof(be));

	return std::make_shared<GameMD>(gIdx, gs, tc, inc, we, be);
}


