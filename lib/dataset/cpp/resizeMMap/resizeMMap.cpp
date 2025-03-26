#include <string>
#include <filesystem>
#include <iostream>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"

namespace fs = std::filesystem;

void resize(fs::path &p, int64_t newsize) {
	if (fs::file_size(p) > newsize) {
		fs::resize_file(p, newsize);
	} else {
		std::cout << "Skipping " << p << " because it is already <= target size" << std::endl;
	}
}

void resizeAll(std::string dn, int64_t ngames, int64_t nmoves) {
	fs::path p(dn+"/gamestarts.npy");
	resize(p, ngames*sizeof(int64_t));
	p = fs::path(dn+"/welos.npy");
	resize(p, ngames*sizeof(int16_t));
	p = fs::path(dn+"/belos.npy");
	resize(p, ngames*sizeof(int16_t));
	p = fs::path(dn+"/mvids.npy");
	resize(p, nmoves*sizeof(int16_t));
	p = fs::path(dn+"/clk.npy");
	resize(p, nmoves*sizeof(int16_t));
	p = fs::path(dn+"/timeCtl.npy");
	resize(p, ngames*sizeof(int16_t));
	p = fs::path(dn+"/inc.npy");
	resize(p, ngames*sizeof(int16_t));
}
	

ABSL_FLAG(std::string, blockDir, "", "directory of block data to be resized");
ABSL_FLAG(int64_t, ngames, 0, "number of actual games in data");
ABSL_FLAG(int64_t, nmoves, 0, "number of actual moves in data");

int main(int argc, char*argv[]) {
	absl::ParseCommandLine(argc, argv);
	std::string blockDir = absl::GetFlag(FLAGS_blockDir);
	int64_t ngames = absl::GetFlag(FLAGS_ngames);
	int64_t nmoves = absl::GetFlag(FLAGS_nmoves);
	if (ngames == 0 || nmoves == 0) {
		std::cout << "ngames or nmoves is 0; aborting..." << std::endl;
		return 0;
	}
	resizeAll(blockDir, ngames, nmoves);
	return 0;
}


