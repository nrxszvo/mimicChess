#include <string>
#include <vector>
#include <tuple>

std::tuple<int, std::vector<std::string>, long > matchNextMove(std::string& moveStr, int idx, int curmv);

struct State {
	std::string weloStr;
	std::string beloStr;
	int welo;
	int belo;
	int time;
	bool validTerm;
	std::string moveStr;
	State(): weloStr(""), beloStr(""), welo(0), belo(0), time(0), validTerm(false), moveStr("") {};
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

std::tuple<std::vector<int>, std::vector<int>, int, int, long > parseMoves(std::string moveStr);

class PgnProcessor {
public:
	PgnProcessor();
	std::string processLine(std::string& line);
	int getWelo();
	int getBelo();
	std::string getMoveStr();
	int getTime();
private:
	State state;
	bool reinit;
};