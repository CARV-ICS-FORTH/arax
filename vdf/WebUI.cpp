#include "WebUI.h"
#include "Misc.h"
#include <vine_pipe.h>
#include <conf.h>
#include <Poco/URI.h>
#include <iostream>
#include <fstream>
#include <random>
#include "Pallete.h"

using namespace Poco;
using namespace Poco::Net;

#define ID_OUT {for(int cnt = 0 ; cnt < id_lvl ; cnt++)out << '\t';}out

extern vine_pipe_s *vpipe;

const char * normalize(const char * label,size_t size)
{
	static char buff[1024];
	snprintf(buff,sizeof(buff),"<tr><th>%s</th><td>%s</td></tr>\n",label,autoRange(size,bytes_to_orders,1024).c_str());
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
	heads << "<tr><th style='border: none;'></th><th class='btn1' onClick=\"toggleSlice(this,'slice" <<bar_count<< "_" << 0 << "')\" style='background-color:#" << Pallete::get(0,15) << "'>";

	char * s = breakdown->heads;
	int parts = 0;
	while(*s)
	{
		if(*s == ',')
		{
			heads << "</th><th class='btn1' onClick=\"toggleSlice(this,'slice" <<bar_count<< "_" << parts+1 << "')\" style='background-color:#" << Pallete::get(parts+1,15) << "'>";
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
			percs << "<td>" << ((int)(1000*perc))/1000.0 << " <div class=u>%</div></td>";
			addBarSlice(bar,bar_count,part,breakdown->part[part],head_vec[part]);
			raw_vals << "<td>" << breakdown->part[part] << "<div class=u>ns</div></td>";
		}
		addBarSlice(bar,bar_count,parts+1,breakdown->part[BREAKDOWN_PARTS+1],"Interarival");
		percs << "<td>" << 100 << "%</td><td class=invisible></td>";
		raw_vals << "<td>" << breakdown->part[BREAKDOWN_PARTS] << "<div class=u>ns</div></td><td class=invisible></td>";
		heads << "<td>" << autoRange(breakdown->part[BREAKDOWN_PARTS]/(float)samples,ns_to_secs,1000) << "</td><td class=invisible></td>";
		percs << "<td>" << (100.0*breakdown->part[BREAKDOWN_PARTS+1])/breakdown->part[BREAKDOWN_PARTS] << "%</td></tr>\n";
		raw_vals << "<td>" << breakdown->part[BREAKDOWN_PARTS+1] << "<div class=u>ns</div></td></tr>\n";
		heads << "<td>" << autoRange(breakdown->part[BREAKDOWN_PARTS+1]/(float)samples,ns_to_secs,1000) << "</td></tr>\n";
	}
	bar << "</div>\n";

	bar_count++;

	return bar.str()+total_bar.str()+"<table style='align-self: center;'>\n"+heads.str()+percs.str()+raw_vals.str()+"</table>\n";
}
#endif

struct allocation
{
	void * name;
	size_t start;
	size_t end;
	size_t size;
	int partition;
};

std::ostream & operator<<(std::ostream & os, const struct allocation & alloc)
{

}

void inspector(void * start, void * end, size_t size, void* arg)
{
	std::vector<allocation> * alloc_vec = (std::vector<allocation> *)arg;
	allocation alloc = {start,(size_t)start,(size_t)end,size};
	alloc_vec->push_back(alloc);
}

void WebUI :: handleRequest(HTTPServerRequest & request,HTTPServerResponse & response)
{
	int id_lvl = 0;
	int type;
	utils_list_s *list;
	utils_list_node_s *itr;
	vine_object_s *obj;
	response.setStatus(HTTPResponse::HTTP_OK);
	response.setContentType("text/html");
	URI uri(request.getURI());

	std::ostream& out = response.send();

	if(!args["embed"])
	{
		ID_OUT << "<!DOCTYPE html>\n";
		ID_OUT << "<html>\n";
		id_lvl++;
		ID_OUT << "<head>\n";
		id_lvl++;
		ID_OUT << "<title>VineWatch</title>\n";

		if(uri.getPath() == "/reset")
		{
			void * temp;
			std::istringstream iss(uri.getQuery());
			iss >> std::hex >> temp;
			if(temp)
				utils_breakdown_init_stats((utils_breakdown_stats_s*)temp);
			ID_OUT << "<meta http-equiv=\"refresh\" content=\"0; url=/\" />";
		}
		ID_OUT << "<link href=\"https://fonts.googleapis.com/css?family=Audiowide\" rel=\"stylesheet\">\n";
		id_lvl--;
		ID_OUT << "</head>\n";
		ID_OUT << "<body>\n";
		id_lvl++;
	}

	std::string src_path = __FILE__;

	src_path.resize(src_path.size()-9);

	ID_OUT << "<style>";
	ID_OUT << "\n" << std::ifstream(src_path+"style.css").rdbuf();
	ID_OUT << "</style>\n";


	ID_OUT << "<script>";
	ID_OUT << "\n" << std::ifstream(src_path+"script.js").rdbuf();
	ID_OUT << "</script>\n";

	ID_OUT << "<div class=version>" << VINE_TALK_GIT_REV << "</div>\n";

	ID_OUT << std::ifstream(src_path+"logo.svg").rdbuf();

	if(!args["noalloc"])
	{
		arch_alloc_stats_s stats = arch_alloc_stats(&(vpipe->allocator));
		ID_OUT << "<h2 onClick=blockTogle('alloc_block')>Allocations</h2>\n";
		ID_OUT << "<div class=block name=alloc_block>\n";
		id_lvl++;
		ID_OUT << "<div class=hgroup>\n";
		id_lvl++;
		ID_OUT << "<div class=hgroup>\n";
		id_lvl++;
		ID_OUT << "<table>\n";
		id_lvl++;
		ID_OUT << _TR(_TH("All Partitions","colspan=2")) << std::endl;
		ID_OUT << "<tr><th>Base</th><td>" << vpipe << "</td></tr>\n";
		ID_OUT <<_TR(_TH("Partitions")+_TD(std::to_string(stats.mspaces)));
		ID_OUT << normalize("Space",stats.total_bytes);
		ID_OUT << normalize("Used",stats.used_bytes);
		ID_OUT << normalize("Free",stats.total_bytes-stats.used_bytes);
		#ifdef ALLOC_STATS
		ID_OUT <<_TR(_TH("Failed allocations")+_TD(std::to_string(stats.allocs[0])));
		ID_OUT <<_TR(_TH("Good allocations")+_TD(std::to_string(stats.allocs[1])));
		ID_OUT <<_TR(_TH("Total Alloc")+_TD(std::to_string(stats.allocs[0]+stats.allocs[1])));
		ID_OUT <<_TR(_TH("Total Free")+_TD(std::to_string(stats.frees)));
		#endif
		id_lvl--;
		ID_OUT << "</table>\n";
		id_lvl--;
		ID_OUT << "</div>\n";

		stats.mspaces = 0;

		std::vector<allocation> allocs;
		std::map<int,std::vector<allocation>> alloc_map;

		arch_alloc_inspect(&(vpipe->allocator),inspector,&allocs);

		size_t base = (size_t)((&(vpipe->allocator))+1);

		for(auto alloc : allocs)
		{
			alloc.start -= base;
			alloc.end -= base;
			alloc.partition = alloc.start/(512*1024*1024);
			alloc_map[alloc.partition].push_back(alloc);
		}
		allocs.clear();

		ID_OUT << "<div class='hgroup greedy'>\n";
		id_lvl++;

		int part = 0;
		do
		{
			stats = arch_alloc_mspace_stats(&(vpipe->allocator),stats.mspaces);
			if(stats.mspaces)
			{
				ID_OUT << "<div class='vgroup bg" << part%2 << "'>\n";
				id_lvl++;
				ID_OUT << "<table>\n";
				id_lvl++;
				//ID_OUT << "<tr><th colspan=2>Partition:" << stats.mspaces << "</th></tr>\n";
				ID_OUT << _TR(_TH("Partition:"+std::to_string(stats.mspaces),"colspan=2")) << std::endl;
				ID_OUT <<normalize("Space",stats.total_bytes);
				ID_OUT <<normalize("Used",stats.used_bytes);
				ID_OUT <<normalize("Free",stats.total_bytes-stats.used_bytes);
				ID_OUT << "</table>\n";
				id_lvl--;

				ID_OUT << "<table>\n";
				id_lvl++;
				ID_OUT << _TR(_TH("Allocations ["+std::to_string(alloc_map[part].size())+"]","colspan=3")) << std::endl;
				ID_OUT << _TR(_TH("Start")+_TH("End")+_TH("Used")) << std::endl;
				for(allocation itr : alloc_map[part])
				{
					ID_OUT << "<tr onmouseover=\"highlight_same(this)\" name=\"alloc" << itr.name << "\">";
					ID_OUT << _TD(std::to_string(itr.start)) + _TD(std::to_string(itr.end)) + _TD(std::to_string(itr.size));
					ID_OUT << "</tr>" << std::endl;
				}
				id_lvl--;
				ID_OUT << "</table>\n";
				id_lvl--;
				ID_OUT << "</div>\n";
			}
			part++;
		}
		while(stats.mspaces);
		id_lvl--;
		ID_OUT << "</div>\n";
		id_lvl--;
		ID_OUT << "</div>\n";
		id_lvl--;
		ID_OUT << "</div>\n";


	}

	if(!args["noobj"])
	{
		ID_OUT << "<h2 onClick=blockTogle('obj_block')>Objects</h2>\n";
		ID_OUT << "<div class=block name=obj_block>\n";
		id_lvl++;

		const char * typestr[VINE_TYPE_COUNT] =
		{
			"Phys Accel",
			"Virt Accel",
			"Vine Procs",
			"Vine Datas"
		};

		ID_OUT << "<div class=hgroup>\n";
		id_lvl++;
		for(type = 0 ; type < VINE_TYPE_COUNT ; type++)
		{
			list = vine_object_list_lock(&(vpipe->objs),(vine_object_type_e)type);
			ID_OUT << "<div class='bg" << type%2 << "'>\n";
			id_lvl++;
			ID_OUT << "<table>\n";
			id_lvl++;
			ID_OUT << _TR(_TH(std::string(typestr[type])+"["+std::to_string(list->length)+"]","colspan=3")) << std::endl;
			ID_OUT << _TR(_TH("Address")+_TH("Name")+_TH("Type")) << std::endl;
			if(list->length)
			{
				utils_list_for_each(*list,itr)
				{
					obj = (vine_object_s*)itr->owner;
					ID_OUT << "<tr onmouseover=\"highlight_same(this)\" name=\"alloc"<< obj << "\"><td>" << obj << "</td>";
					switch(type)
					{
						case VINE_TYPE_PHYS_ACCEL:
							ID_OUT << _TD(obj->name) << _TD(vine_accel_type_to_str(((vine_accel_s*)obj)->type));
							break;
						case VINE_TYPE_VIRT_ACCEL:
							ID_OUT << _TD(obj->name) << _TD(vine_accel_type_to_str(((vine_accel_s*)obj)->type));
							break;
						case VINE_TYPE_PROC:
							ID_OUT << _TD(obj->name) << _TD(vine_accel_type_to_str(((vine_proc_s*)obj)->type));
							break;
						case VINE_TYPE_DATA:
							ID_OUT << _TD("Data") << _TD(std::to_string(((vine_data_s*)obj)->size));
							break;
						default:
							ID_OUT << _TD(obj->name) << _TD("Unknown");
							break;
					}
					ID_OUT << "<tr>\n";
				}
			}
			else
			{
				ID_OUT << _TR(_TD(std::string("No ")+typestr[type],"colspan=3")) << std::endl;
			}
			id_lvl--;
			ID_OUT << "</table>\n";
			vine_object_list_unlock(&(vpipe->objs),(vine_object_type_e)type);
			id_lvl++;
			ID_OUT << "</div>\n";


		}
		id_lvl--;
		ID_OUT << "</div>\n";
		id_lvl--;
		ID_OUT << "</div>\n";

	}

	#ifdef BREAKS_ENABLE
	if(!args["nobreak"])
	{
		bool had_breaks = false;
		ID_OUT << "<h2 onClick=blockTogle('brk_block')>Breakdowns</h2>\n";
		ID_OUT << "<div class=block name=brk_block>\n";
		id_lvl++;
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
		id_lvl--;
		ID_OUT << "</div>\n";
	}

	if(!args["notelemetry"])
	{
		ID_OUT << "<h2 onClick=blockTogle('tlm_block')>Telemetry</h2>\n";
		ID_OUT << "<div class=block name=tlm_block>\n";
		id_lvl++;
		collector->rawDump(out);
		id_lvl--;
		ID_OUT << "</div>\n";
	}
	#endif

	if(!args["embed"])
	{
		id_lvl--;
		ID_OUT << "</body>\n";
		id_lvl--;
		ID_OUT << "</html>\n";
	}
	out.flush();
}

WebUI :: WebUI(std::map<std::string,bool> & args,Collector * collector)
: collector(collector), args(args)
{
	gethostname(hostname,1024);
}
