#include <string>
#include <vector>
#include <tuple>

std::tuple<int, std::vector<std::string> > matchNextMove(std::string& moveStr, int idx, int curmv, bool requireClk);

struct State {
	std::string weloStr;
	std::string beloStr;
	int welo;
	int belo;
	int time;
	int inc;
	bool validTerm;
	std::string moveStr;
	State(): weloStr(""), beloStr(""), welo(0), belo(0), time(0), inc(0), validTerm(false), moveStr("") {};
	void init() {
		this->weloStr = "";
		this->beloStr = "";
		this->welo = 0;
		this->belo = 0;
		this->time = 0;
		this->validTerm = false;
		this->moveStr = "";
	};
};

std::tuple<std::vector<int16_t>, std::vector<int16_t> > parseMoves(std::string moveStr, bool requireClk=true);

class PgnProcessor {
public:
	PgnProcessor(int minSec, int maxSec, int maxInc);
	std::string processLine(std::string& line);
	int getWelo();
	int getBelo();
	std::string getMoveStr();
	int getTime();
	int getInc();
private:
	State state;
	bool reinit;
	int minSec;
	int maxSec;
	int maxInc;
};
