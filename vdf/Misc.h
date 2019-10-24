#ifndef VDF_MISC_HEADER
	#define VDF_MISC_HEADER
	#include <sstream>
	#include <string>

	static const char * bytes_to_orders[] = {"b ","Kb","Mb","Gb","Tb","Pb","Eb","Zb",0};
	static const char * ns_to_secs[] = {"ns","us","ms","s","KiloSec","MegaSec",0};



	std::string autoRange(size_t value,const char * units[],size_t order,size_t precission = 1000);

	#define autoBytes(VALUE) autoRange(VALUE,bytes_to_orders,1024)

	#define autoNs(VALUE) autoRange(VALUE,bytes_to_orders,1024)

	std::string tag_gen(std::string tag,std::string inner_html = "",std::string attrs = "");

	#define _S(VAL) std::to_string(VAL)
	#define _TR(...) tag_gen("tr",__VA_ARGS__)
	#define _TD(...) tag_gen("td",__VA_ARGS__)
	#define _TH(...) tag_gen("th",__VA_ARGS__)
	#define _RECT(FILL,X,Y,W,H,ATTR)									\
							tag_gen("rect","",((FILL!="")?("fill=#"+std::string(FILL)):(std::string()))+		\
							" x="+_S(X)+						\
							" y="+_S(Y)+						\
							" width="+_S(W)+					\
							" height="+_S(H)+					\
							ATTR)
	#define _TEXT(...) tag_gen("text",__VA_ARGS__)

#endif
