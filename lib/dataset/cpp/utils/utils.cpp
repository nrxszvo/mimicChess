#include "utils.h"

std::string zfill(int time) {
	std::string timeStr = std::to_string(time);
	if (timeStr.size() == 1) timeStr = "0" + timeStr;
	return timeStr;
}


std::pair<std::string, tp> getEta(uintmax_t total, uintmax_t soFar, tp &start) {	
	auto stop = hrc::now();
	if (soFar == 0) {
		return std::make_pair("tbd", stop);
	}
	soFar = std::min(total, soFar);
	long ellapsed = std::chrono::duration_cast<milli>(stop-start).count();
	long remaining_ms = (total-soFar) * ellapsed / soFar;
	int remaining = int(remaining_ms/1e3);
	int hrs = remaining / 3600;
	int minutes = (remaining % 3600) / 60;
	int secs = remaining % 60;
	std::string etaStr = std::to_string(hrs) + ":" + zfill(minutes) + ":" + zfill(secs);
	return std::make_pair(etaStr, stop);
}

std::string getEllapsedStr(int ellapsed) {
	int hrs = ellapsed/3600;	
	int minutes = (ellapsed % 3600) / 60;
	int secs = ellapsed % 60;
	return std::to_string(hrs) + ":" + zfill(minutes) + ":" + zfill(secs);
}

std::string getEllapsedStr(tp& start, tp& stop) {
	int ellapsed = std::chrono::duration_cast<std::chrono::seconds>(stop-start).count();
	return getEllapsedStr(ellapsed);
}

bool ellapsedGTE(tp& last, int thresh) {
	auto now = hrc::now();
	int ellapsed = std::chrono::duration_cast<std::chrono::seconds>(now-last).count();
	return ellapsed >= thresh;
}

	
