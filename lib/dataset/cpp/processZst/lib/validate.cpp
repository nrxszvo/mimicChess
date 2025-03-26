#include <string>
#include "inference.h"
#include "parseMoves.h"

using namespace std;

const char PIECE_TO_NAME[] = {
	'R',
	'N',
	'B',
	'Q',
	'K',
	'B',
	'N',
	'R',
	'P',
	'P',
	'P',
	'P',
	'P',
	'P',
	'P',
	'P',
	'R',
	'N',
	'B',
	'Q',
	'K',
	'B',
	'N',
	'R',
	'P',
	'P',
	'P',
	'P',
	'P',
	'P',
	'P',
	'P'
};

const char INT_TO_FILE[] = {
	'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'
};

char intToRow(int r) {
	return '1' + r;
}

string decodeMvid(int mvid) {	
	if (mvid == QCASTLEW || mvid == QCASTLEB) {
		return "O-O-O";
	} else if (mvid == KCASTLEW || mvid == KCASTLEB) {
		return "O-O";
	} else {
		char piece = PIECE_TO_NAME[mvid/64];
		int cell = mvid % 64;
		int r = cell / 8;
		int f = cell % 8;
		string mv = string() + INT_TO_FILE[f] + intToRow(r);
		return piece + mv;
	}
}

bool compareMoves(string& mv, string& pfr) {
	if (mv == pfr) {
		return true;
	}
	char piece = pfr[0];
	char file = pfr[1];
	char rank = pfr[2];
	if (piece == 'P') {
		return mv.find(pfr.substr(1,2)) != string::npos;
	} else {
		return mv.find(piece) != string::npos && mv.find(file) != string::npos && mv.find(rank) != string::npos;
	}
}

vector<pair<int, string> > validateGame(int gameid, string moveStr, vector<int16_t>& mvids, bool requireClk) {
	vector<pair<int,string> > results;
	int curmv = 1;
	int mvIdx = 0;
	int idIdx = 0;
	vector<string> matches;
	long re_ns;
	while (mvIdx < moveStr.size()) {
		tie(mvIdx, matches) = matchNextMove(moveStr, mvIdx, curmv, requireClk); 
		if (mvIdx == moveStr.size() && matches.size() == 0) {
			break;
		}
		for (int i=0; i<matches.size(); i+=2) {
			int mvid = mvids[idIdx++];
			string pfr = decodeMvid(mvid);
			if (!compareMoves(matches[i], pfr)) {
				string err = "Move mismatch: game " + to_string(gameid) + ", move " + to_string(curmv) + ", " + matches[i] + " != " + pfr;
				results.push_back(make_pair(gameid, err));
			}
		}
		curmv++;
	}
	return results;
}
