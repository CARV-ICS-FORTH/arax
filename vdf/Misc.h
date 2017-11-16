#ifndef VDF_MISC_HEADER
	#define VDF_MISC_HEADER
	#include <sstream>
	#include <string>

	static const char * bytes_to_orders[] = {"b ","Kb","Mb","Gb","Tb","Pb",0};
	static const char * ns_to_secs[] = {"ns","us","ms","s","KiloSec","MegaSec",0};



	std::string autoRange(size_t value,const char * units[],int order,int precission = 1000);

	std::string tag_gen(std::string tag,std::string inner_html = "",std::string attrs = "");

	#define _TR(...) tag_gen("tr",__VA_ARGS__)
	#define _TD(...) tag_gen("td",__VA_ARGS__)
	#define _TH(...) tag_gen("th",__VA_ARGS__)
#endif
