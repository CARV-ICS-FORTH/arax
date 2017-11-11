#include "Misc.h"

std::string autoRange(size_t value,const char * units[],int order)
{
	int c = 0;
	float ret = value;
	std::ostringstream oss;
	while(ret >= order && units[c])
	{
		ret /= (float)order;
		c++;
	}
	oss << ((int)(ret*1000))/1000.0 << " " << units[c];
	return oss.str();
}

std::string tag_gen(std::string tag,std::string inner_html,std::string attrs)
{
	std::ostringstream oss;

	oss << "<" << tag << " " << attrs << ">" << inner_html << "</" << tag << ">";

	return oss.str();
}
