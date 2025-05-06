#include <algorithm>
#include <vector>
#include <string>
#include <re2/re2.h>
#include <tuple>
#include "parseMoves.h"
#include "profiling/profiler.h"

using namespace std;

const string MV_PAT = "(O-O-O\\+?#?|O-O\\+?#?|[a-hRNBQK]+[0-9=x]*[a-hRNBQK]*[0-9]*[=RNBQ+#]*)\\??\\??\\!?";
const string CLK_PAT = "(?:\\{ (?:\\[%eval [0-9.\\-#]+\\] )?\\[%clk ([0-9:]+)\\] \\})?";
const string ALT_LINE_PAT = "(?:\\([0-9]+\\.[0-9a-hRNBQK]*\\))?";
const string NUM_PAT = "[0-9]+\\.";
const string NUM_ALT_PAT = "(?:" + NUM_PAT + "\\.\\.)?";
const string RESULT_PAT = ".*(1/2-1/2|0-1|1-0)";

const re2::RE2 twoMoves(NUM_PAT + " " + MV_PAT + " " + CLK_PAT + " ?" + NUM_ALT_PAT + " ?" + "(?:" + MV_PAT + ")? ?" + CLK_PAT);

string movenoToStr(int moveno) {
	return to_string(moveno) + ". ";
}

string nextMoveStr(string& moveStr, int& idx, int curmv) {
	int mvstart = idx;
	string nextmv = movenoToStr(curmv+1);

	bool inParens = false;
	bool inBracket = false;
	while(idx < moveStr.size()) {
		if (inParens) {
			while (moveStr[idx] != ')') idx++;
			inParens = false;
		} else if (inBracket) {
			while (moveStr[idx] != '}') idx++;
			inBracket = false;
		} else if (moveStr[idx] == '(') {
			inParens = true;
		} else if (moveStr[idx] == '{') {
			inBracket = true;
		} else if (moveStr.substr(idx, nextmv.size()) == nextmv) {
			break;
		}
		idx++;
	}

	string ss = moveStr.substr(mvstart, idx-mvstart);
	return ss;
}

string clkToSec(string timeStr) {
	int m = stoi(timeStr.substr(2, 2));
	int s = stoi(timeStr.substr(5, 2));
	return to_string(m * 60 + s);
}

tuple<string, string, int8_t> parseMoves(string moveStr) {
	string mvs = "";
	string clk = "";
	int8_t result;
	int curmv = 1;
	int idx = 0;
	vector<string> matches;
	
	while (idx < moveStr.size()) {
		string ss = nextMoveStr(moveStr, idx, curmv);
		profiler.start("regex");
		string wm="", bm="", wclk="", bclk="";
		re2::RE2::PartialMatch(ss, twoMoves, &wm, &wclk, &bm, &bclk);
		profiler.stop("regex");

		if (idx == moveStr.size()) {
			string res;
			re2::RE2::PartialMatch(moveStr.substr(idx-7,7), RESULT_PAT, &res);
			if (res == "1/2-1/2") {
				result = 2;
			} else if (res == "0-1") {
				result = 1;
			} else {
				result = 0;
			}
		}

		if (wm == "") {
			break;
		} 
		if (wclk != "") {
			clk += clkToSec(wclk) + " ";
		}
		mvs += wm + " ";
		if (bclk != "") {
			clk += clkToSec(bclk) + " ";
		}
		if (bm != "") {
			mvs += bm + " ";
		}
		curmv++;
	}
	return make_tuple(mvs.substr(0, mvs.size()-1), clk.substr(0, clk.size()-1), result);
}

const string TERM_PATS[] = {
	"Normal",
	"Time forfeit"
};

const re2::RE2 timeRe("\\[TimeControl \"([0-9]+)\\+*([0-9]+)\"\\]");
const re2::RE2 termRe("\\[Termination \"(.+)\"\\]");
const re2::RE2 reW("\\[WhiteElo \"([0-9]+)\"\\]");
const re2::RE2 reB("\\[BlackElo \"([0-9]+)\"\\]");

string processRawLine(string& line, State& state, int minSec, int maxSec, int maxInc) {
	line.erase(remove(line.begin(), line.end(), '\n'), line.cend());
	if (line.size() > 0) {
		if (line[0] == '[') {
			if (line.substr(0, 6) == "[Event") {
				state.init();
			} else if (line.substr(0, 9) == "[WhiteElo") {
				state.weloStr = line;
			} else if (line.substr(0,9) == "[BlackElo") {
				state.beloStr = line;
			} else if (line.substr(0,12) == "[TimeControl") {
				int tim, inc = 0;
				if (re2::RE2::PartialMatch(line, timeRe, &tim, &inc)) {
					if (inc <= maxInc && tim <= maxSec && tim >= minSec) {
						state.time = tim;
						state.inc = inc;
					}
				}
			} else if (line.substr(0, 12) == "[Termination") {
				string term;
				re2::RE2::PartialMatch(line, termRe, &term);
				for (auto tp: TERM_PATS) {
					if (term.find(tp) != string::npos) {
						state.validTerm = true;
						break;
					}
				}
			}
		} else if (line[0] == '1') {
			if (state.time > 0 && state.weloStr != "" && state.beloStr != "") {
				int welo, belo;
				bool haveW = re2::RE2::PartialMatch(state.weloStr, reW, &welo);
				bool haveB = re2::RE2::PartialMatch(state.beloStr, reB, &belo);
				if (haveW && haveB) {
					state.welo = welo;
					state.belo = belo;
					state.moveStr = line;
					return "COMPLETE";
				}
			}
			return "INVALID";
		}
	}
	return "INCOMPLETE";
}

PgnProcessor::PgnProcessor(int minSec, int maxSec, int maxInc): reinit(false), minSec(minSec), maxSec(maxSec), maxInc(maxInc) {}

string PgnProcessor::processLine(string& line) {
	if (this->reinit) {
		this->state.init();
		this->reinit = false;
	}
	string code = processRawLine(line, this->state, this->minSec, this->maxSec, this->maxInc);
	if (code == "COMPLETE" || code == "INVALID") {
		this->reinit = true;
	}
	return code;
}
int PgnProcessor::getWelo() {
	return this->state.welo;
}
int PgnProcessor::getBelo() {
	return this->state.belo;
}
string PgnProcessor::getMoveStr() {
	return this->state.moveStr;
}
int PgnProcessor::getTime() {
	return this->state.time;
}
int PgnProcessor::getInc() {
	return this->state.inc;
}
