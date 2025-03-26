#include <chrono>
#include <string>

using hrc = std::chrono::high_resolution_clock;
using tp = std::chrono::time_point<std::chrono::high_resolution_clock>;
using milli = std::chrono::milliseconds;

std::pair<std::string, tp> getEta(uintmax_t total, uintmax_t soFar, tp &start);
std::string zfill(int time);
std::string getEllapsedStr(tp& start, tp& stop);
std::string getEllapsedStr(int ellapsed);
bool ellapsedGTE(tp& last, int thresh);
