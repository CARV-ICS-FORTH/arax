#include <map>
#include <iomanip>
#include <iostream>
#include <vine_pipe.h>
#include <core/vine_data.h>

int main(int argc, char * argv[])
{
	std::map<size_t,std::map<std::string,size_t>> leaks;

	vine_pipe_s * vpipe = vine_talk_init();

	int type;
	size_t leaks_cnt = 0;
	utils_list_s *list;
	utils_list_node_s *itr;
	vine_data_s *obj;

	list = vine_object_list_lock(&(vpipe->objs),VINE_TYPE_DATA);

	utils_list_for_each(*list,itr)
	{
		obj = (vine_data_s*)itr->owner;
		leaks[vine_data_size(obj)][obj->obj.name]++;
		leaks_cnt++;
	}

	vine_object_list_unlock(&(vpipe->objs),VINE_TYPE_DATA);

	std::cerr << "Found " << leaks_cnt << " data leaks from " << leaks.size() << " sources:\n";

	for(auto & leak_sz : leaks)
	{
		for( auto leak_n : leak_sz.second)
		{
			std::cerr.width(12);
			std::cerr << leak_n.second << " leak" << ((leak_n.second>1)?"s":"") << " of ";
			std::cerr.width(12);
			std::cerr << leak_sz.first << " bytes from ";
			std::cerr.width(VINE_OBJECT_NAME_SIZE);
			std::cerr << leak_n.first << " (total: ";
			std::cerr.width(6);
			std::cerr << std::setprecision(3) << std::fixed << (leak_n.second*leak_sz.first)/(1024*1024.0) << " Mb)\n";
		}
	}

	#ifndef VINE_DATA_ANNOTATE
	std::cerr << "Warning: VINE_DATA_ANNOTATE not enabled, leaks will be anonymous!\n";
	#endif

	vine_talk_exit();
	return 0;
}
