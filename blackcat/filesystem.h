#ifndef BLACKCAT_FILESYSTEMS_H_
#define BLACKCAT_FILESYSTEMS_H_

#include <sys/types.h>
#include <sys/stat.h>
#include <string>
#include <fstream>
#include "string.h"

namespace bc {
namespace filesystem {

#if defined _WIN32 || defined __CYGWIN__
#define BC_FILE_SEPERATOR '\\'
#else
#define BC_FILE_SEPERATOR '/'
#endif

static constexpr char separator = BC_FILE_SEPERATOR;

inline bool directory_exists(const std::string& name) {
#ifdef _MSC_VER
	struct stat info;
	stat(name.c_str(), &info);
	return info.st_mode & S_IFDIR;
#else
	return system(("test -d " + name).c_str()) == 0;
#endif
}

inline int mkdir(const std::string& name) {
	return system(("mkdir " + name).c_str()); 
}

inline bool file_exists(const std::string& name) {
	struct stat buffer;
	return stat(name.c_str(), &buffer) == 0;
}

inline bc::string make_path(const bc::string& path) {
	return path;
}

template<class... Strs>
inline bc::string make_path(const bc::string& str, const Strs&... strs) {
	bc::string right_path = make_path(strs...);
	bool lsep = str.endswith(separator);
	bool rsep = right_path.startswith(separator);

	if (lsep != rsep)
		return str + right_path;
	else if (lsep && rsep)
		return str.substr(0, str.size()-2) + right_path;
	else
		return str + BC_FILE_SEPERATOR + right_path;
}

}


}

#endif
