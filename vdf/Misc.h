#ifndef VDF_MISC_HEADER
	#define VDF_MISC_HEADER
	#include <sstream>
	#include <string>

	std::string autoRange(size_t value,const char * units[],int order);

	std::string tag_gen(std::string tag,std::string inner_html = "",std::string attrs = "");

	#define _TR(...) tag_gen("tr",__VA_ARGS__)
	#define _TD(...) tag_gen("td",__VA_ARGS__)
	#define _TH(...) tag_gen("th",__VA_ARGS__)
#endif
