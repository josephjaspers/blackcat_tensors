#ifndef BLACKCAT_FILESYSTEMS_H_
#define BLACKCAT_FILESYSTEMS_H_

#include <sys/types.h>
#include <sys/stat.h>
#include <string>

#include "BlackCat_String.h"

namespace BC {
namespace filesystem {

#if defined _WIN32 || defined __CYGWIN__
#define BC_FILE_SEPERATOR '\\'
#else
#define BC_FILE_SEPERATOR '/'
#endif

static constexpr char separator = BC_FILE_SEPERATOR;

inline bool directory_exists(const std::string& name) {
	struct stat info;
	stat(name.c_str(), &info);
	return info.st_mode & S_IFDIR;
}

inline int mkdir(const std::string& name) {
	return system(("mkdir " + name).c_str()); 
}

inline bool file_exists(const std::string& name) {
	struct stat buffer;
	return stat(name.c_str(), &buffer) == 0;
}

inline BC::string make_path(const BC::string& path) {
	return path;
}

template<class... Strs>
inline BC::string make_path(const BC::string& str, const Strs&... strs) {
	BC::string right_path = make_path(strs...);
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
