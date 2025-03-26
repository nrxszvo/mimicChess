#include <string>
#include <vector>
#include <functional>
#include <stdexcept>
#include <memory>
#include "inference.h"

using namespace std;

int sign(int v) {
	return v > 0 ? 1 : v < 0 ? -1 : 0;
}

tuple<vector<vector<std::shared_ptr<Piece> > >, vector<std::shared_ptr<Piece> >, vector<std::shared_ptr<Piece> > > initBoardState() {
	vector<vector<std::shared_ptr<Piece> > > board(8, vector<std::shared_ptr<Piece>>(8));
	vector<std::shared_ptr<Piece> > white, black;

	for (int i=0; i<8; i++) {
		board[1][i] = std::make_shared<Piece>('P', 1, i, 8+i, COLORW);
		board[6][i] = std::make_shared<Piece>('P', 6, i, 24+i, COLORB);
	}
	board[0][QROOK] = std::make_shared<Piece>('R', 0, 0, 0, COLORW);
	board[0][QKNIGHT] = std::make_shared<Piece>('N', 0, 1, 1, COLORW);
	board[0][QBISHOP] = std::make_shared<Piece>('B', 0, 2, 2, COLORW);
	board[0][QUEEN] = std::make_shared<Piece>('Q', 0, 3, 3, COLORW);
	board[0][KING] = std::make_shared<Piece>('K', 0, 4, 4, COLORW);
	board[0][KBISHOP] = std::make_shared<Piece>('B', 0, 5, 5, COLORW);
	board[0][KKNIGHT] = std::make_shared<Piece>('N', 0, 6, 6, COLORW);
	board[0][KROOK] = std::make_shared<Piece>('R', 0, 7, 7, COLORW);
	board[7][QROOK] = std::make_shared<Piece>('R', 7, 0, 16, COLORB);
	board[7][QKNIGHT] = std::make_shared<Piece>('N', 7, 1, 17, COLORB);
	board[7][QBISHOP] = std::make_shared<Piece>('B', 7, 2, 18, COLORB);
	board[7][QUEEN] = std::make_shared<Piece>('Q', 7, 3, 19, COLORB);
	board[7][KING] = std::make_shared<Piece>('K', 7, 4, 20, COLORB);
	board[7][KBISHOP] = std::make_shared<Piece>('B', 7, 5, 21, COLORB);
	board[7][KKNIGHT] = std::make_shared<Piece>('N', 7, 6, 22, COLORB);
	board[7][KROOK] = std::make_shared<Piece>('R', 7, 7, 23, COLORB);

	for (int i=0; i<2; i++) {
		for (int j=0; j<8; j++) {
			white.push_back(board[i][j]);
			black.push_back(board[7-i][j]);
		}
	}
	return make_tuple(board, white, black);
}

int fileToInt(char file) {
	return file - 'a';
}

int rankToInt(char rank) {
	return (rank - '1');
}

int cellToInt(string cell) {
	return fileToInt(cell[0]) + 8 * rankToInt(cell[1]);
}

void parseRankOrFile(char rorf, MoveState& state) {
	if (rorf >= 'a' && rorf <= 'h') {
		state.hasFile = true;
		state.srcFile = rorf;
	} else {
		state.hasRank = true;
		state.srcRank = rorf;
	}
}

int ctoi(char c) {
	return c - '0';
}

MoveState parsePawnMove(string mv, string last_mv, int color) {
	MoveState ret;
	ret.piece = 'P';
	if (mv.size() == 2) {
		ret.dest = mv;
	} else {
		if (mv[1] == 'x') {
			parseRankOrFile(mv[0], ret);
			ret.dest = mv.substr(2,2);
			if (last_mv.size() == 2 && last_mv[0] == mv[2] && ctoi(last_mv[1]) + color == ctoi(mv[3])) {
				ret.enpassant = true;
			}
			if (mv.size() > 4 && mv[4] == '=') {
				ret.promotion = true;
				ret.piece = mv[5];
			}
		} else if (mv[2] == '=') {
			ret.dest = mv.substr(0,2);
			ret.src = mv[0] + to_string(ctoi(mv[1])-color);
			ret.piece = mv[3];
			ret.promotion = true;
		} else if (mv[1] >= '1' && mv[1] <= '8') {
			ret.src = mv.substr(0,2);
			ret.dest = mv.substr(3,2);
		} else {
			throw runtime_error("Pawn parse error: " + mv);
		}
	}
	return ret;
}

MoveState parseNonPawnMove(string mv) {
	MoveState ret;
	ret.piece = mv[0];
	if (mv.size() == 3) {
		ret.dest = mv.substr(1,2);
	} else if (mv.size() == 4) {
		if (mv[1] != 'x') {
			parseRankOrFile(mv[1], ret);
		}
		ret.dest = mv.substr(2, 2);
	} else if (mv.size() == 5) {
		if (mv[2] == 'x') {
			parseRankOrFile(mv[1], ret);
			ret.dest = mv.substr(3, 2);
		} else {
			ret.src = mv.substr(1,2);
			ret.dest = mv.substr(3,2);
		}
	} else {
		if (mv.size() != 6) throw runtime_error("unexpected move length: " + to_string(mv.size()));
		ret.src = mv.substr(1,2);
		ret.dest = mv.substr(4,2);
	}
	return ret;
}

MoveState parseMove(string mv, string lastMv, int color) {

	if (mv == "O-O") {
		MoveState ret;
		ret.castle = "king";
		return ret;
	} else if (mv == "O-O-O") {
		MoveState ret;
		ret.castle = "queee";
		return ret;
	} else if (mv[0] >= 'a' && mv[0] <= 'h') {
		return parsePawnMove(mv, lastMv, color);
	} else {
		return parseNonPawnMove(mv);
	}
}

int castleToMvid(MoveState &ms, vector<vector<std::shared_ptr<Piece> > >& board, vector<std::shared_ptr<Piece> >& state) {
	if (ms.castle == "king") {
		state[KING]->file = 6;
		state[KROOK]->file = 5;

		if (state[0]->color == COLORW) {
			board[0][4] = nullptr;
			board[0][6] = state[KING];
			board[0][7] = nullptr;
			board[0][5] = state[KROOK];
			return KCASTLEW;
		} else {
			board[7][4] = nullptr;
			board[7][6] = state[KING];
			board[7][7] = nullptr;
			board[7][5] = state[KROOK];
			return KCASTLEB;
		}
	} else {
		state[KING]->file = 2;
		state[QROOK]->file = 3;
		if (state[0]->color == COLORW) {
			board[0][4] = nullptr;
			board[0][2] = state[KING];
			board[0][0] = nullptr;
			board[0][3] = state[QROOK];
			return QCASTLEW;
		} else {
			board[7][4] = nullptr;
			board[7][2] = state[KING];
			board[7][0] = nullptr;
			board[7][3] = state[QROOK];
			return QCASTLEB;
		}
	}
}

bool legalPawnMove(std::shared_ptr<Piece> piece, vector<vector<std::shared_ptr<Piece> > >& board, int dr, int df, bool enpassant) {
	auto [sr, sf] = piece->pos();
	if (sf == df) {
		if (abs(dr-sr) > 2) {
			return false;
		}
		if (abs(dr-sr) == 2) {
			if (piece->color == COLORW && (sr != 1 || dr != 3)) {
				return false;
			}
			if (piece->color == COLORB && (sr != 6 || dr != 4)) {
				return false;
			}
			if (board[dr-piece->color][df] != nullptr) {
				return false;
			}
		} else if (piece->color * (dr-sr) != 1) {
			return false;
		}
		return board[dr][df] == nullptr;
	} else if (abs(df-sf) == 1) {
		if (piece->color*(dr-sr) != 1) {
			return false;
		}
		if (enpassant) {
			return board[sr][df] != nullptr && board[sr][df]->color == -piece->color;
		} else {
			return board[dr][df] != nullptr && board[dr][df]->color == -piece->color;
		}
	} else {
		return false;
	}
}

bool legalRookMove(std::shared_ptr<Piece> piece, vector<vector<std::shared_ptr<Piece> > >& board, int dr, int df) {
	auto [sr, sf] = piece->pos();
	if (sf != df && sr != dr) {
		return false;
	}
	if (sf != df) {
		for (int i = sf + sign(df-sf); i != df; i += sign(df-sf)) {
			if (board[dr][i] != nullptr) {
				return false;
			}
		}
	} else {
		for (int i = sr + sign(dr-sr); i != dr; i += sign(dr-sr)) {
			if (board[i][df] != nullptr) {
				return false;
			}
		}
	}
	return board[dr][df] == nullptr || board[dr][df]->color != piece->color;
}

bool legalKnightMove(std::shared_ptr<Piece> piece, vector<vector<std::shared_ptr<Piece>> >& board, int dr, int df) {
	auto [sr, sf] = piece->pos();
	if (abs(sf-df) == 2) {
		if (abs(sr-dr) != 1) {
			return false;
		}
	} else if (abs(sr-dr) == 2) {
		if (abs(sf-df) != 1) {
			return false;
		}
	} else {
		return false;
	}
	return board[dr][df] == nullptr || board[dr][df]->color != piece->color;
}

bool legalBishopMove(std::shared_ptr<Piece> piece, vector<vector<std::shared_ptr<Piece>> >& board, int dr, int df) {
	auto [sr, sf] = piece->pos();
	if (abs(sf-df) != abs(sr-dr)) {
		return false;
	}

	int finc = sign(df-sf);
	int rinc = sign(dr-sr);
	for (int i = 1; i < abs(sf-df); i++) {
		int r = sr + rinc*i;
		int f = sf + finc*i;
		if (board[r][f] != nullptr) {
			return false;
		}
	}
	return board[dr][df] == nullptr || board[dr][df]->color != piece->color;

}

bool legalQueenMove(std::shared_ptr<Piece> piece, vector<vector<std::shared_ptr<Piece>> >& board, int dr, int df) {
	return legalRookMove(piece, board, dr, df) || legalBishopMove(piece, board, dr, df);
}

bool legalMove(std::shared_ptr<Piece> piece, vector<vector<std::shared_ptr<Piece>> >& board, string& dest, bool enpassant = false) {
	int dr = rankToInt(dest[1]);
	int df = fileToInt(dest[0]);
	switch (piece->name) {
		case 'P':
			return legalPawnMove(piece, board, dr, df, enpassant);
		case 'R':
			return legalRookMove(piece, board, dr, df);
		case 'N':
			return legalKnightMove(piece, board, dr, df);
		case 'B':
			return legalBishopMove(piece, board, dr, df);
		case 'Q':
			return legalQueenMove(piece, board, dr, df);
		default:
			return true;
	}
}

bool attacking(std::shared_ptr<Piece> a, std::shared_ptr<Piece> b, vector<vector<std::shared_ptr<Piece>> >& board) {
	auto [dr, df] = b->pos();
	switch (a->name) {
		case 'P':
			{
				auto [sr, sf] = a->pos();
				return a->color * (dr-sr) == 1 && abs(df-sf) == 1;
			}
		case 'R':
			return legalRookMove(a, board, dr, df);
		case 'N':
			return legalKnightMove(a, board, dr, df);
		case 'B':
			return legalBishopMove(a, board, dr, df);
		case 'Q':
			return legalQueenMove(a, board, dr, df);
		default:
			return false;
	}
}

bool kingInCheck(vector<vector<std::shared_ptr<Piece>> >& board, vector<std::shared_ptr<Piece>>& cur, vector<std::shared_ptr<Piece>>& opp) {
	std::shared_ptr<Piece> king = cur[KING];	
	for (auto piece: opp) {
		if (!piece->captured && attacking(piece, king, board)) {
			return true;
		}
	}
	return false;
}


pair<int, int> mvToPos(string& mv) {
	int r = rankToInt(mv[1]);
	int f = fileToInt(mv[0]);
	return make_pair(r,f);
}

bool srcInferenceMatch(std::shared_ptr<Piece> piece, MoveState& mvdata) {
	auto [sr, sf] = mvToPos(mvdata.src);
	if (piece->rank == sr && piece->file == sf) {
		if (mvdata.promotion) {
			if (piece->name == 'P') {
				piece->name = mvdata.piece;
				return true;
			}
		} else {
			return true;
		}
	}
	return false;
}

bool srcRfInferenceMatch(std::shared_ptr<Piece> piece, MoveState& mvdata, vector<vector<std::shared_ptr<Piece>> >& board) {
	bool srcCond;
	if (mvdata.hasFile) {
		srcCond = fileToInt(mvdata.srcFile) == piece->file;	
	} else {
		srcCond = rankToInt(mvdata.srcRank) == piece->rank;
	}
	if (board[piece->rank][piece->file] == piece 
			&& (piece->name == mvdata.piece || (mvdata.promotion && piece->name == 'P'))
			&& srcCond && 
			legalMove(piece, board, mvdata.dest, mvdata.enpassant)) {
		if (mvdata.promotion) {
			if (piece->name == 'P') {
				piece->name = mvdata.piece;
				return true;
			}
		} else {
			return true;
		}
	}
	return false;
}

bool genericInferenceMatch(std::shared_ptr<Piece> piece, MoveState& mvdata, vector<vector<std::shared_ptr<Piece>> >& board) {
	return board[piece->rank][piece->file] == piece && piece->name == mvdata.piece && legalMove(piece, board, mvdata.dest);
}

int16_t inferMvid(MoveState& mvdata, vector<vector<std::shared_ptr<Piece>> >& board, vector<std::shared_ptr<Piece>>& state, vector<std::shared_ptr<Piece>> oppState) {
	if (mvdata.castle != "") {
		return castleToMvid(mvdata, board, state);
	}

	vector<std::shared_ptr<Piece>> candidates;
	
	auto srcIm = [&](std::shared_ptr<Piece> piece) { return srcInferenceMatch(piece, mvdata); };
	auto srcRfIm = [&](std::shared_ptr<Piece> piece) { return srcRfInferenceMatch(piece, mvdata, board); };
	auto genIm = [&](std::shared_ptr<Piece> piece) { return genericInferenceMatch(piece, mvdata, board); };
	
	function<bool(std::shared_ptr<Piece>)> cond;
	if (mvdata.src != "") {
		cond = srcIm;
	} else if (mvdata.srcRank != '\0' || mvdata.srcFile != '\0') {
		cond = srcRfIm;
	} else {
		cond = genIm;
	}
	
	for (auto piece: state) {
		if (!piece->captured && cond(piece)) {
			candidates.push_back(piece);
		}
	}

	auto [dr, df] = mvToPos(mvdata.dest);

	auto updateState = [&](std::shared_ptr<Piece> piece, bool enpassant) {
		auto [sr, sf] = piece->pos();	
		piece->rank = dr;
		piece->file = df;
		board[sr][sf] = nullptr;
		std::shared_ptr<Piece> tmp;
		if (enpassant) {
			int tr = dr - piece->color;
			tmp = board[tr][df];
			board[tr][df] = nullptr;
		} else {
			tmp = board[dr][df];
		}
		if (tmp) {
			tmp->captured = true;
		}
		board[dr][df] = piece;
		return make_tuple(sr, sf, tmp);
	};

	auto revertState = [&](std::shared_ptr<Piece> piece, int sr, int sf, std::shared_ptr<Piece> tmp, bool enpassant) {
		board[sr][sf] = piece;
		if (tmp) {
			tmp->captured = false;
			board[tmp->rank][tmp->file] = tmp;
		}
		if (tmp == nullptr || enpassant) {
			board[dr][df] = nullptr;
		}
		piece->rank = sr;
		piece->file = sf;
	};

	std::shared_ptr<Piece> piece;
	if (candidates.size() > 1) {
		vector<std::shared_ptr<Piece>> valid;
		for (auto piece: candidates) {
			auto [sr, sf, tmp] = updateState(piece, mvdata.enpassant);
			if (!kingInCheck(board, state, oppState)) {
				valid.push_back(piece);
			}
			revertState(piece, sr, sf, tmp, mvdata.enpassant);
		}
		if (valid.size() != 1) throw runtime_error("could not resolve possible moves");
		piece = valid[0];
	} else {
		piece = candidates[0];
	}
	updateState(piece, mvdata.enpassant);
	int mvid = piece->pid * 64 + cellToInt(mvdata.dest);
	return (int16_t)mvid;
}

MoveParser::MoveParser() {
	tie(this->board, this->white, this->black) = initBoardState();
	this->prevMv = "";
	this->color = COLORW;
}


int16_t MoveParser::inferId(string& mv) {
	MoveState mvdata = parseMove(mv, this->prevMv, this->color);
	vector<std::shared_ptr<Piece> > state, other;
	if (this->color == COLORW) {
		state = this->white;
		other = this->black;
	} else {
		state = this->black;
		other = this->white;
	}
	this->prevMv = mv;
	this->color = -this->color;
	return inferMvid(mvdata, this->board, state, other);
}
