#include "zstd.h"
#include <fstream>
#include <ios>
#include <sstream>
#include <vector>

std::vector<size_t> getFrameBoundaries(std::string zst, int nBoundaries);

class DecompressStream {	
	ZSTD_DCtx* dctx;
	ZSTD_inBuffer in;
	ZSTD_outBuffer out;
	char* in_mem;
	char* out_mem;
	std::ifstream infile;
	std::stringstream ss;
	size_t zstdRet;
	size_t frameSize;
	size_t maxBytes;
	size_t totalRead;
	std::string rem;
public:
	DecompressStream(std::string zstfn, size_t frameStart, size_t frameEnd, size_t frameSize=1024*1024);
	~DecompressStream();
	std::streamsize decompressFrame();
	void getLines(std::vector<std::string>& lines);
	size_t getFrameSize();
	float getProgress();
};	
