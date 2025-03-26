#include <string>
#include <vector>
#include <memory>

#define COLORW 1
#define COLORB -1
#define QROOK 0
#define QKNIGHT 1
#define QBISHOP 2
#define QUEEN 3
#define KING 4
#define KBISHOP 5
#define KKNIGHT 6
#define KROOK 7

#define KCASTLEW QBISHOP*64+7
#define KCASTLEB (16+QBISHOP)*64 + 63
#define QCASTLEW KBISHOP*64
#define QCASTLEB (16+KBISHOP)*64 + 56

#define NOOP QBISHOP*64+3

struct MoveState {
	char piece;
	std::string src;
	bool hasFile;
	char srcFile;
	bool hasRank;
	char srcRank;
	std::string dest;
	bool enpassant;
	bool promotion;
	std::string castle;
	MoveState(): piece('\0'), src(""), hasFile(false), srcFile('\0'), hasRank(false), srcRank('\0'), dest(""), enpassant(false), promotion(false), castle("") {};
};	

struct Piece {
	char name;
	int rank;
	int file;
	int pid;
	int color;
	bool captured;
	Piece(char name, int rank, int file, int pid, int color): name(name), rank(rank), file(file), pid(pid), color(color), captured(false) {};
	std::pair<int,int> pos() { return std::make_pair(this->rank, this->file); };
};

class MoveParser {
public:
	MoveParser(); 
	int16_t inferId(std::string& mv);
private:
	std::vector<std::vector<std::shared_ptr<Piece>> > board;
	std::vector<std::shared_ptr<Piece> > white;
	std::vector<std::shared_ptr<Piece> > black;
	std::string prevMv;
	int color;
};

