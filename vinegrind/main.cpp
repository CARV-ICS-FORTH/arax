#include <map>
#include <climits>
#include <iomanip>
#include <iostream>
#include <vine_pipe.h>
#include <core/vine_data.h>

size_t getSizeOfVineObject(vine_object_s & obj)
{
	switch(obj.type)
	{
		case VINE_TYPE_PHYS_ACCEL:	/* Physical Accelerator */
			return sizeof(vine_accel_s);
		case VINE_TYPE_VIRT_ACCEL:	/* Virtual Accelerator */
			return sizeof(vine_vaccel_s);
		case VINE_TYPE_PROC:			/* Procedure */
			return sizeof(vine_proc_s);
		case VINE_TYPE_DATA:			/* Data Allocation */
			return sizeof(vine_data_s) + vine_data_size((vine_data_s*)&obj);
		case VINE_TYPE_TASK:			/* Task */
			return sizeof(vine_task_msg_s);
		default:
			throw std::runtime_error("Encountered unknown object type ("+std::to_string(obj.type)+")");
	};
}

std::string getNameOfVineObject(vine_object_s & obj)
{
	if(obj.type == VINE_TYPE_TASK)
	{
		vine_task_msg_s * task = (vine_task_msg_s * )&obj;

		if(!task->proc)
			return "NULL PROC TASK";

		return ((vine_proc_s*)(task->proc))->obj.name;
	}
	else
		return obj.name;
}

std::string printSize(size_t size)
{
#define POWER_SIZE 6
	const char * unit[POWER_SIZE] =
	{
		" B",
		"KB",
		"MB",
		"GB",
		"PB",
		"TB"
	};
	int power = 0;
	while( size >= 1024*10 && power < POWER_SIZE )
	{
		size /= 1024;
		power++;
	}
	return std::to_string(size)+unit[power];
}

class Leak
{
	friend std::ostream & operator<<(std::ostream & os,const Leak & leak);
	public:
		Leak()
		: leaks(0), size(0), max_ref(0), min_ref(INT_MAX)
		{
		}
		void track(vine_object_s & obj)
		{
			leaks++;
			name = getNameOfVineObject(obj);
			size = getSizeOfVineObject(obj);
			total += size;
			max_ref = std::max(max_ref,vine_object_refs(&obj));
			min_ref = std::min(min_ref,vine_object_refs(&obj));
		}
	private:
		std::string name;
		uint64_t leaks;
		size_t size;
		size_t total;
		int max_ref;
		int min_ref;
};

std::ostream & operator<<(std::ostream & os,const Leak & leak)
{
	std::cerr.width(12);
	std::cerr << leak.leaks << " leak" << ((leak.leaks>1)?"s":" ") << " of ";
	std::cerr.width(12);
	std::cerr << printSize(leak.size)<< " from ";
	std::cerr.width(VINE_OBJECT_NAME_SIZE);
	std::cerr << leak.name << " (total: ";
	std::cerr.width(6);
	std::cerr << std::setprecision(5)
		<< std::fixed << printSize(leak.total)
		<< ", refs: [" << leak.min_ref << " , "
		<< leak.max_ref << "] )" << std::endl;

	return os;
}

void leak_check(vine_pipe_s * vpipe,vine_object_type_e type,std::string stype)
{
	size_t leaks_cnt = 0;
	size_t leak_total = 0;
	utils_list_s *list;
	utils_list_node_s *itr;
	vine_object_s *obj;
	std::map<size_t,std::map<std::string,Leak>> leaks;

	list = vine_object_list_lock(&(vpipe->objs),type);

	utils_list_for_each(*list,itr)
	{
		obj = (vine_object_s*)itr->owner;
		leaks[getSizeOfVineObject(*obj)][obj->name].track(*obj);
		leaks_cnt++;
		leak_total += getSizeOfVineObject(*obj);
	}

	vine_object_list_unlock(&(vpipe->objs),type);

	std::cerr << "Found " << leaks_cnt << " " << stype << " leaks, totaling " << printSize(leak_total) << ", from " << leaks.size() << " sources:\n";

	for(auto & leak_sz : leaks)
	{
		for( auto leak_n : leak_sz.second)
		{
			std::cerr << leak_n.second << std::endl;
		}
	}

}

int main(int argc, char * argv[])
{
	vine_pipe_s * vpipe = vine_talk_init();

	leak_check(vpipe,VINE_TYPE_DATA,"data");
	leak_check(vpipe,VINE_TYPE_TASK,"task");

	#ifndef VINE_DATA_ANNOTATE
	std::cerr << "Warning: VINE_DATA_ANNOTATE not enabled, leaks will be anonymous!\n";
	#endif

	vine_talk_exit();
	return 0;
}
