#include "WebUI.h"
#include "Misc.h"
#include <vine_pipe.h>
#include "core/vine_data.h"
#include <conf.h>
#include <Poco/URI.h>
#include <Poco/Path.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include "Pallete.h"

using namespace Poco;
using namespace Poco::Net;

#define ID_OUT out << id_str
#define ID_INC id_str+='\t'
#define ID_DEC id_str.resize(id_str.size()-1)

extern vine_pipe_s *vpipe;

const char * normalize(const char * label,size_t size)
{
	static char buff[1024];
	snprintf(buff,sizeof(buff),"<tr><th>%s</th><td>%s</td></tr>",label,autoBytes(size).c_str());
	return buff;
}


int bar_count = 0;

char hostname[1024];

#ifdef BREAKS_ENABLE

void addBarSlice(std::ostream & os,int bar_count,int id,unsigned long long value,std::string title)
{
	os << "<div id='slice"<<bar_count<< "_" << id <<"' title='" << title << "' class=slice1 style = 'flex-grow:" << value << ";background-color:#" << Pallete::get(id,15) << ";'></div>\n";
}

std::string generateBreakBar(std::ostream & out,utils_breakdown_stats_s * breakdown)
{
	int samples = breakdown->samples;
	std::ostringstream bar;
	std::ostringstream heads;
	std::ostringstream percs;
	std::ostringstream raw_vals;
	std::ostringstream total_bar;
	std::ostringstream head_str;
	std::vector<std::string> head_vec;
	bar << "<div class=bar>\n";
	total_bar << "<div class=tot_bar>\n";
	heads << "<tr><th style='border: none;'></th>\n<th class='btn1' onClick=\"toggleSlice(this,'slice" <<bar_count<< "_" << 0 << "')\" style='background-color:#" << Pallete::get(0,15) << "'>";

	char * s = breakdown->heads;
	int parts = 0;
	while(*s)
	{
		if(*s == ',')
		{
			heads << "</th>\n<th class='btn1' onClick=\"toggleSlice(this,'slice" <<bar_count<< "_" << parts+1 << "')\" style='background-color:#" << Pallete::get(parts+1,15) << "'>";
			parts++;
			head_vec.push_back(head_str.str());
			head_str.str("");
			head_str.clear();
		}
		else
		{
			if(*s == '_')
			{
				heads << ' ';
				head_str << ' ';
			}
			else
			{
				heads << *s;
				head_str << *s;
			}
		}
		s++;
	}

	addBarSlice(total_bar,bar_count,parts,breakdown->part[BREAKDOWN_PARTS],"Total");
	total_bar << "</div>\n";

	if(breakdown->part[BREAKDOWN_PARTS])
	{
		heads << "Total</th><th class=invisible></th><th onClick=\"toggleSlice(this,'slice" <<bar_count<< "_" << parts+1 << "')\" style='background-color:#" << Pallete::get(parts+1,15) << "'> Interarival</th></tr>\n";
		heads << "<tr><th>Time</th>";
		percs << "<tr><th>Percent</th>";
		raw_vals << "<tr><th>Time(ns)</th>";
		for(int part = 0 ; part < parts ; part++)
		{
			float perc = (100.0*breakdown->part[part])/breakdown->part[BREAKDOWN_PARTS];
			heads << "<td>" << autoRange(breakdown->part[part]/(float)samples,ns_to_secs,1000) << "</td>";
			percs << "<td>" << ((int)(1000*perc))/1000.0 << " <div class=u>&#37;</div></td>";
			addBarSlice(bar,bar_count,part,breakdown->part[part],head_vec[part]);
			raw_vals << "<td>" << breakdown->part[part] << "<div class=u>ns</div></td>";
		}
		addBarSlice(bar,bar_count,parts+1,breakdown->part[BREAKDOWN_PARTS+1],"Interarival");
		percs << "<td>" << 100 << "&#37;</td><td class=invisible></td>";
		raw_vals << "<td>" << breakdown->part[BREAKDOWN_PARTS] << "<div class=u>ns</div></td><td class=invisible></td>";
		heads << "<td>" << autoRange(breakdown->part[BREAKDOWN_PARTS]/(float)samples,ns_to_secs,1000) << "</td><td class=invisible></td>";
		percs << "<td>" << (100.0*breakdown->part[BREAKDOWN_PARTS+1])/breakdown->part[BREAKDOWN_PARTS] << "&#37;</td></tr>\n";
		raw_vals << "<td>" << breakdown->part[BREAKDOWN_PARTS+1] << "<div class=u>ns</div></td></tr>\n";
		heads << "<td>" << autoRange(breakdown->part[BREAKDOWN_PARTS+1]/(float)samples,ns_to_secs,1000) << "</td></tr>\n";
	}
	bar << "</div>\n";

	bar_count++;

	return bar.str()+total_bar.str()+"<table style='align-self: center;'>\n"+heads.str()+percs.str()+raw_vals.str()+"</table>\n";
}
#endif

std::string minPtr(void * ptr,int digits=2)
{
	std::ostringstream oss;
	oss << ptr;
	return oss.str().substr(digits);
}

int calcDigits(void * ptr,size_t size)
{
	std::ostringstream iss0,iss1;
	iss0 << ptr;
	iss1 << (void*)(((uint8_t*)ptr)+size);

	std::string a = iss0.str(),b = iss1.str();
	int l;

	for(l = 0 ; l < std::min(a.size(),b.size()) ; l++)
	{
		if(a[l] != b[l])
			break;
	}

	return l;
}

struct allocation
{
	void * name;
	size_t start;
	size_t end;
	size_t used;
	int partition;
};

std::ostream & operator<<(std::ostream & os, const struct allocation & alloc)
{
	int digits = calcDigits(vpipe,vpipe->shm_size);
	int64_t space = alloc.end - alloc.start;
	std::string us;
	if(space == alloc.used)
		us = _TD(_S(space),"colspan=2");
	else
		us = _TD(_S(alloc.used) + _TD(_S(space)));
	os << "<tr onmouseover=\"highlight_same(this)\" name=\"alloc" << minPtr(alloc.name,digits) << "\">"
	<< _TD(_S(alloc.start) + " - " + _S(alloc.end)) + us
	<< "</tr>" << std::endl;
	return os;
}

void inspector(void * start, void * end, size_t used, void* arg)
{
	std::vector<allocation> * alloc_vec = (std::vector<allocation> *)arg;
	if(used)
		used -= sizeof(size_t);
	allocation alloc = {start,(size_t)start,(size_t)end,used};
	alloc_vec->push_back(alloc);
}

template<class T>
std::string getAcceleratorType(T * obj)
{
	return _TD(vine_accel_type_to_str(obj->type));
}

void printThrotle(std::ostream & out,std::string & id_str,vine_throttle_s * th,std::string name)
{
	size_t a = vine_throttle_get_available_size(th);
	size_t t = vine_throttle_get_total_size(th);
	ID_OUT << "<table>\n";
	ID_INC;
	ID_OUT << "<tr><th>" << name << "</th></tr>\n";
	ID_OUT << "<tr><td class='nopad'><div class='throt_bar' style='width:" << ((t-a)*100)/t<< "%''></td></tr>\n";
	ID_OUT << "<tr><td>" << autoBytes(t-a) << '/' << autoBytes(t) << "</td></tr>\n";
	ID_DEC;
	ID_OUT << "</table>\n";
}

void WebUI :: handleRequest(HTTPServerRequest & request,HTTPServerResponse & response)
{
	std::string id_str = "";
	int type;
	utils_list_s *list;
	utils_list_node_s *itr;
	vine_object_s *obj;
	response.setStatus(HTTPResponse::HTTP_OK);
	response.setContentType("text/html");
	URI uri(request.getURI());

	std::ostream& out = response.send();

	int digits = calcDigits(vpipe,vpipe->shm_size);

	if(!args["embed"])
	{
		ID_OUT << "<!DOCTYPE html>\n";
		ID_OUT << "<html>\n";
		ID_INC;
		ID_OUT << "<head>\n";
		ID_INC;
		ID_OUT << "<title>VineWatch</title>\n";
		ID_OUT << "<link href=\"https://fonts.googleapis.com/css?family=Roboto\" rel=\"stylesheet\">\n";
		ID_DEC;
		ID_OUT << "</head>\n";
		ID_OUT << "<body>\n";
		ID_INC;
	}

	std::string src_path = __FILE__;

	src_path.resize(src_path.size()-9);

	ID_OUT << "<style>";
	ID_OUT << "\n" << std::ifstream(src_path+"style.css").rdbuf();
	ID_OUT << "</style>\n";


	ID_OUT << "<script>";
	ID_OUT << "\n" << std::ifstream(src_path+"script.js").rdbuf();
	ID_OUT << "</script>\n";

	ID_OUT << "<div class=version>" << VINE_TALK_GIT_REV << " - " << VINE_TALK_GIT_BRANCH << "</div>\n";

	ID_OUT << std::ifstream(src_path+"logo.svg").rdbuf();

	if(!args["noconf"])
	{
		ID_OUT << "<h2 onClick=blockTogle('conf_block')>Config</h2>\n";
		ID_OUT << "<div class=block name=conf_block >\n";
		ID_INC;
		ID_OUT << "<table>\n";
		ID_INC;
		ID_OUT << _TR(_TH("Key")+_TH("Value")) << std::endl;


		std::ifstream cfg(Poco::Path::expand(VINE_CONFIG_FILE));

		if(!cfg)
			ID_OUT << _TR(_TH("File")+_TD(Poco::Path::expand(VINE_CONFIG_FILE)+"(NotFound!)")) << std::endl;
		else
			ID_OUT << _TR(_TH("File")+_TD(Poco::Path::expand(VINE_CONFIG_FILE))) << std::endl;

		std::ostringstream iss;

		iss << (void*)vpipe;

		ID_OUT << _TR(_TH("Base")+_TD(iss.str())) << std::endl;
		ID_OUT << _TR(normalize("Size",vpipe->shm_size)) << std::endl;

		ID_OUT << _TR(_TH("")+_TH("")) << std::endl;
		do
		{
			std::string key,value;
			cfg >> key >> value;
			if(cfg)
				ID_OUT << _TR(_TH(key)+_TD(value)) << std::endl;
		}
		while(cfg);

		ID_DEC;
		ID_OUT << "</table>\n";
		ID_DEC;
		ID_OUT << "</div>\n";
	}

	if(!args["nosizes"])
	{
		ID_OUT << "<h2 onClick=blockTogle('size_block')>Struct Sizes</h2>\n";
		ID_OUT << "<div class=block name=size_block>\n";
		ID_INC;
		ID_OUT << "<table>\n";
		ID_INC;
		ID_OUT << _TR(_TH("Type")+_TH("Size")) << std::endl;
		#define TYPE_SIZE(TYPE) \
			ID_OUT << _TR(_TH(#TYPE)+_TD(std::to_string(sizeof(TYPE))+" B")) << std::endl
		TYPE_SIZE(vine_proc_s);
		TYPE_SIZE(vine_accel_s);
		TYPE_SIZE(vine_data_s);
		TYPE_SIZE(vine_task_msg_s);
		TYPE_SIZE(vine_pipe_s);
		TYPE_SIZE(utils_queue_s);
		TYPE_SIZE(vine_vaccel_s);
		#undef TYPE_SIZE
		ID_DEC;
		ID_OUT << "</table>\n";
		ID_DEC;
		ID_OUT << "</div>\n";
	}

	if(!args["nothrot"])
	{
		ID_OUT << "<h2 onClick=blockTogle('throt_block')>Throttling</h2>\n";
		ID_OUT << "<div class=block name=throt_block>\n";
		ID_INC;
		ID_OUT << "<div>";
		list = vine_object_list_lock(&(vpipe->objs),VINE_TYPE_PHYS_ACCEL);
		size_t p_cnt = list->length;
		if(p_cnt)
		{
			ID_INC;
			ID_OUT << "<div class='hgroup'>";
			utils_list_for_each(*list,itr)
			{
				auto p = (vine_accel_s*)itr->owner;
				printThrotle(out,id_str,&(p->throttle),p->obj.name);
			}
			ID_DEC;
			ID_OUT << "</div>\n";
		}
		vine_object_list_unlock(&(vpipe->objs),VINE_TYPE_PHYS_ACCEL);
		ID_INC;
		ID_OUT << "<div class='hgroup'>";
		printThrotle(out,id_str,&(vpipe->throttle),"Shm");
		ID_DEC;
		ID_OUT << "</div>\n";
		ID_DEC;
		ID_OUT << "</div>\n";
		ID_DEC;
		ID_OUT << "</div>\n";
	}

	if(!args["noalloc"])
	{
		arch_alloc_stats_s stats = arch_alloc_stats(&(vpipe->allocator));
		ID_OUT << "<h2 onClick=blockTogle('alloc_block')>Allocations</h2>\n";
		ID_OUT << "<div class=block name=alloc_block>\n";
		ID_INC;
		ID_OUT << "<div class=vgroup>\n";
		ID_INC;
		ID_OUT << "<div class=hgroup>\n";
		ID_INC;
		ID_OUT << "<table>\n";
		ID_INC;
		ID_OUT << _TR(_TH("All Partitions","colspan=2")) << std::endl;
		ID_OUT << "<tr><th>Base</th><td>" << vpipe << "</td></tr>\n";
		ID_OUT <<_TR(_TH("Partitions")+_TD(_S(stats.mspaces))) << std::endl;
		ID_OUT << normalize("Space",stats.total_bytes) << std::endl;
		ID_OUT << normalize("Used",stats.used_bytes) << std::endl;
		ID_OUT << normalize("Free",stats.total_bytes-stats.used_bytes) << std::endl;
		#ifdef ALLOC_STATS
		ID_OUT <<_TR(_TH("Failed allocations")+_TD(_S(stats.allocs[0]))) << std::endl;
		ID_OUT <<_TR(_TH("Good allocations")+_TD(_S(stats.allocs[1]))) << std::endl;
		auto total_allocs = stats.allocs[0]+stats.allocs[1];
		ID_OUT <<_TR(_TH("Total Alloc")+_TD(_S(total_allocs))) << std::endl;
		ID_OUT <<_TR(_TH("Total Free")+_TD(_S(stats.frees))) << std::endl;
		auto leaks = stats.allocs[1]-stats.frees;
		ID_OUT <<_TR(_TH("Leaks")+_TD(_S(leaks) + "(" + _S((leaks*100)/total_allocs) + "&#37;)")) << std::endl;
		#endif
		ID_DEC;
		ID_OUT << "</table>\n";
		ID_DEC;
		ID_OUT << "</div>\n";

		stats.mspaces = 0;

		std::vector<allocation> allocs;
		std::map<int,std::vector<allocation>> alloc_map;

		#ifdef ALLOC_STATS
		allocs.reserve(stats.allocs[1]);	// Optional Optimization
		#endif

		arch_alloc_inspect(&(vpipe->allocator),inspector,&allocs);

		size_t base = (size_t)((&(vpipe->allocator))+1);

		for(auto alloc : allocs)
		{
			alloc.start -= base;
			alloc.end -= base;
			alloc.partition = alloc.start/(ALLOC_PART_MB*1024*1024ul);
			alloc_map[alloc.partition].push_back(alloc);
		}
		allocs.clear();

		ID_OUT << "<div class='hgroup greedy'>\n";
		ID_INC;

		int part = 0;
		do
		{
			stats = arch_alloc_mspace_stats(&(vpipe->allocator),stats.mspaces);
			if(stats.mspaces)
			{
				ID_OUT << "<div class='vgroup bg" << part%2 << "'>\n";
				ID_INC;
				ID_OUT << "<table>\n";
				ID_INC;
				//ID_OUT << "<tr><th colspan=2>Partition:" << stats.mspaces << "</th></tr>\n";
				ID_OUT << _TR(_TH("Partition:"+_S(stats.mspaces),"colspan=2")) << std::endl;
				ID_OUT <<normalize("Space",stats.total_bytes) << std::endl;
				ID_OUT <<normalize("Used",stats.used_bytes) << std::endl;
				ID_OUT <<normalize("Free",stats.total_bytes-stats.used_bytes) << std::endl;
				ID_DEC;
				ID_OUT << "</table>\n";


				ID_OUT << "<table>\n";
				ID_INC;
				ID_OUT << _TR(_TH("Allocations ["+_S(alloc_map[part].size())+"]","colspan=4")) << std::endl;
				ID_OUT << _TR(_TH("Range")+_TH("Used")+_TH("Reserved")) << std::endl;
				for(allocation itr : alloc_map[part])
					ID_OUT << itr;
				ID_DEC;
				ID_OUT << "</table>\n";
				ID_DEC;
				ID_OUT << "</div>\n";
			}
			part++;
		}
		while(stats.mspaces);
		ID_DEC;
		ID_OUT << "</div>\n";
		ID_DEC;
		ID_OUT << "</div>\n";
		ID_DEC;
		ID_OUT << "</div>\n";
		ID_DEC;
		ID_OUT << "</div>\n";


	}

	if(!args["noobj"])
	{
		ID_OUT << "<h2 onClick=blockTogle('obj_block')>Objects</h2>\n";
		ID_OUT << "<div class=block name=obj_block>\n";
		ID_INC;

		const char * typestr[VINE_TYPE_COUNT] =
		{
			"Phys Accel",
			"Virt Accel",
			"Vine Procs",
			"Vine Datas",
			"Vine Tasks"
		};

		ID_OUT << "<div class=hgroup>\n";
		ID_INC;
		for(type = 0 ; type < VINE_TYPE_COUNT ; type++)
		{
			list = vine_object_list_lock(&(vpipe->objs),(vine_object_type_e)type);
			ID_OUT << "<div class='bg" << type%2 << "'>\n";
			ID_INC;
			ID_OUT << "<table>\n";
			ID_INC;
			ID_OUT << _TR(_TH(std::string(typestr[type])+"["+_S(list->length)+"]","colspan=5")) << std::endl;
			ID_OUT << _TR(_TH("Address")+_TH("Name")+_TH("Refs")+_TH("Type")+_TH("Extra")) << std::endl;
			if(list->length)
			{
				utils_list_for_each(*list,itr)
				{
					obj = (vine_object_s*)itr->owner;
					ID_OUT << "<tr onmouseover=\"highlight_same(this)\" name=\"alloc"<< minPtr(obj,digits) << "\"><th>" << minPtr(obj,digits) << "</th>"<< _TD(obj->name) << _TD(_S(vine_object_refs(obj)));
					switch(type)
					{
						case VINE_TYPE_PHYS_ACCEL:
							ID_OUT << getAcceleratorType((vine_accel_s*)obj) << _TD("Rev:"+_S(vine_accel_get_revision(((vine_accel_s*)obj))));
							break;
						case VINE_TYPE_VIRT_ACCEL:
							ID_OUT << getAcceleratorType((vine_vaccel_s*)obj) << _TD("Queue:"+_S(utils_queue_used_slots(vine_vaccel_queue((vine_vaccel_s*)obj))));
							break;
						case VINE_TYPE_PROC:
							ID_OUT << getAcceleratorType((vine_proc_s*)obj) << _TD("");
							break;
						case VINE_TYPE_DATA:
							ID_OUT << _TD(_S(((vine_data_s*)obj)->size)) << _TD("");
							break;
						case VINE_TYPE_TASK:
						{
							vine_task_msg_s * task = (vine_task_msg_s*)obj;
							ID_OUT << getAcceleratorType(((vine_proc_s*)((task)->proc))) << _TD(((vine_object_s*)((task)->proc))->name);
						}
							break;
						default:
							ID_OUT << _TD("Unknown") << _TD("");
							break;
					}
					ID_OUT << "</tr>\n";
				}
			}
			else
			{
				ID_OUT << _TR(_TH(std::string("No ")+typestr[type],"colspan=5")) << std::endl;
			}
			ID_DEC;
			ID_OUT << "</table>\n";
			vine_object_list_unlock(&(vpipe->objs),(vine_object_type_e)type);
			ID_DEC;
			ID_OUT << "</div>\n";
		}
		ID_DEC;
		ID_OUT << "</div>\n";
		ID_DEC;
		ID_OUT << "</div>\n";

	}

	#ifdef BREAKS_ENABLE
	if(!args["nobreak"])
	{
		bool had_breaks = false;
		ID_OUT << "<h2 onClick=blockTogle('brk_block')>Breakdowns</h2>\n";
		ID_OUT << "<div class=block name=brk_block>\n";
		ID_INC;
		vine_proc_s* proc;
		list = vine_object_list_lock(&(vpipe->objs),VINE_TYPE_PROC);
		utils_list_for_each(*list,itr)
		{
			obj = (vine_object_s*)itr->owner;
			proc = (vine_proc_s*)obj;
			int samples = proc->breakdown.samples;
			if(samples)
			{
				ID_OUT << "<h2>" << vine_accel_type_to_str(proc->type) << "::" << obj->name << "(" << samples << " samples,task average)@" << hostname << ":</h2>\n";
				ID_OUT << "<a href='reset?" << (void*)&(proc->breakdown) << "'>Reset breakdown</a>\n";
				out << generateBreakBar(out,&(proc->breakdown));
				had_breaks = true;
			}
		}
		vine_object_list_unlock(&(vpipe->objs),VINE_TYPE_PROC);
		if(!had_breaks)
		{
			ID_OUT << "No breakdowns collected, run something first!\n";
		}
		ID_DEC;
		ID_OUT << "</div>\n";
	}

	if(!args["notelemetry"])
	{
		ID_OUT << "<h2 onClick=blockTogle('tlm_block')>Telemetry</h2>\n";
		ID_OUT << "<div class=block name=tlm_block>\n";
		ID_INC;
		collector->rawDump(out);
		ID_DEC;
		ID_OUT << "</div>\n";
	}
	#endif

	if(!args["embed"])
	{
		ID_DEC;
		ID_OUT << "</body>\n";
		ID_DEC;
		ID_OUT << "</html>\n";
	}
	out.flush();
}

WebUI :: WebUI(std::map<std::string,bool> & args,Collector * collector)
: collector(collector), args(args)
{
	gethostname(hostname,1024);
}
