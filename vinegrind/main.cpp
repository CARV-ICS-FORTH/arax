#include <iostream>
#include <vine_pipe.h>
#include <map>

struct Metrics
{
	int type;
	size_t leaks;
	size_t size;
};

int main(int argc, char * argv[])
{
	std::map<std::string,Metrics> leaks;

	vine_pipe_s * vpipe = vine_talk_init();

	int type;
	utils_list_s *list;
	utils_list_node_s *itr;
	vine_object_s *obj;

	for(type = 0 ; type < VINE_TYPE_COUNT ; type++)
	{
		list = vine_object_list_lock(&(vpipe->objs),(vine_object_type_e)type);

		utils_list_for_each(*list,itr)
		{
			obj = (vine_object_s*)itr->owner;
			auto & stats = leaks[obj->name];
			stats.leaks++;
		}

		vine_object_list_unlock(&(vpipe->objs),(vine_object_type_e)type);
	}
	std::cerr << "Found leaks from " << leaks.size() << " sources\n";
	for(auto & leak : leaks)
	{
		std::cerr.width(12);
		std::cerr << leak.second.leaks << " leaks of " << leak.first << std::endl;
	}

	vine_talk_exit();
	return 0;
}
