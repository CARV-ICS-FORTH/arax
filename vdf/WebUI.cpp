#include "WebUI.h"
#include "Misc.h"
#include <vine_pipe.h>
#include <Poco/URI.h>

using namespace Poco;
using namespace Poco::Net;

#define ID_OUT {for(int cnt = 0 ; cnt <= id_lvl ; cnt++)out << '\t';}out << '\t'

extern vine_pipe_s *vpipe;

const char * units[] = {"b ","Kb","Mb","Gb","Tb",0};

const char * normalize(const char * label,size_t size,int id_lvl)
{
	static char buff[1024];
	snprintf(buff,sizeof(buff),"%*c<tr><th>%s</th><td>%s</td></tr>\n",id_lvl-1,'\t',label,autoRange(size,units,1024).c_str());
	return buff;
}

void WebUI :: handleRequest(HTTPServerRequest & request,HTTPServerResponse & response)
{
	int id_lvl = 1;
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
		ID_OUT <<	"<!DOCTYPE html>\n"
		"	<html>\n"
		"		<head>\n"
		"			<title>VineWatch</title>\n";

		if(uri.getPath() == "/reset")
		{
			void * temp;
			std::istringstream iss(uri.getQuery());
			iss >> std::hex >> temp;
			if(temp)
				utils_breakdown_init_stats((utils_breakdown_stats_s*)temp);
			ID_OUT << "<meta http-equiv=\"refresh\" content=\"0; url=/\" />";
		}

		ID_OUT <<	"		</head>\n"
		"		<body>\n";
	}

	ID_OUT << "<style>\n";
	ID_OUT << "	h1 {align-self:center;}\n";
	ID_OUT << "	td, th{border: 1px solid black;border-collapse: collapse;}\n";
	ID_OUT << "	table{border-collapse: collapse;margin:10px;display: flex;align-items: center;}\n";
	ID_OUT << "	tr th:first-child{text-align: right;}\n";
	ID_OUT << "	td{text-align: center;}\n";
	ID_OUT << "	body {display:flex;flex-flow: column wrap;}\n";
	ID_OUT << "	.group {flex-flow: row wrap;display:flex;justify-content: center;}\n";
	ID_OUT << "	.bar {width: 90%;height: 3em;display: flex;border: 1px solid;align-self:center;}\n";
	ID_OUT << "	.tot_bar {width: 90%;height: 0.5em;display: flex;border: 1px solid;align-self:center;border-top: none;}\n";
	ID_OUT << "	.slice1 {}\n";
	ID_OUT << "	.slice0 {display:none}\n";
	ID_OUT << "	.btn0 {opacity:0.25;}\n";
	ID_OUT << "	.btn1 {opacity:1;}\n";
	ID_OUT << "	h1 {display:flex;}\n";
	ID_OUT << ".u {-webkit-touch-callout: none;-webkit-user-select: none;-khtml-user-select: none;-moz-user-select: none;-ms-user-select: none;user-select: none;display: inline;}\n";
	ID_OUT << ".invisible {border: none;width: 2em;}\n";
	ID_OUT << "</style>\n";


	ID_OUT << "<script>";
	ID_OUT << "function toggleSlice(btn,slice){";
	ID_OUT << "elem = document.getElementById(slice);";
	ID_OUT << "if(elem.className == 'slice0'){";
	ID_OUT << "elem.className = 'slice1';";
	ID_OUT << "btn.className = 'btn1';";
	ID_OUT << "}";
	ID_OUT << "else{";
	ID_OUT << "elem.className = 'slice0';";
	ID_OUT << "btn.className = 'btn0';";
	ID_OUT << "}";
	ID_OUT << "}";
	ID_OUT << "</script>";


	if(!args["noalloc"])
	{
		arch_alloc_stats_s stats = arch_alloc_stats(&(vpipe->allocator));
		ID_OUT << "<h1>Allocator status</h1>\n";
		ID_OUT << "<div class=group>\n";
		id_lvl++;
		ID_OUT << "<table>\n";
		id_lvl++;
		ID_OUT << "<tr><th colspan=2>All Partitions</th></tr>\n";
		ID_OUT <<normalize("Partitions",stats.mspaces,id_lvl);
		ID_OUT <<normalize("Space",stats.total_bytes,id_lvl);
		ID_OUT <<normalize("Used",stats.used_bytes,id_lvl);
		ID_OUT <<normalize("Free",stats.total_bytes-stats.used_bytes,id_lvl);
		#ifdef ALLOC_STATS
		ID_OUT <<normalize("Failed allocations",stats.allocs[0],id_lvl);
		ID_OUT <<normalize("Good allocations",stats.allocs[1],id_lvl);
		ID_OUT <<normalize("Frees",stats.frees,id_lvl);
		#endif
		id_lvl--;
		ID_OUT << "</table>\n";
		id_lvl--;
		ID_OUT << "</div>\n";

		stats.mspaces = 0;

		ID_OUT << "<div class=group>\n";
		id_lvl++;
		do
		{
			stats = arch_alloc_mspace_stats(&(vpipe->allocator),stats.mspaces);
			if(stats.mspaces)
			{
				ID_OUT << "<table>\n";
				id_lvl++;
				ID_OUT << "<tr><th colspan=2>Partition:" << stats.mspaces << "</th></tr>\n";
				ID_OUT <<normalize("Space",stats.total_bytes,id_lvl);
				ID_OUT <<normalize("Used",stats.used_bytes,id_lvl);
				ID_OUT <<normalize("Free",stats.total_bytes-stats.used_bytes,id_lvl);
				ID_OUT << "</table>\n";
				id_lvl--;
			}
		}
		while(stats.mspaces);
		ID_OUT << "</div>\n";
		id_lvl--;
	}

	if(!args["noobj"])
	{
		ID_OUT << "<h1>Objects status</h1>\n";

		const char * typestr[VINE_TYPE_COUNT] =
		{
			"Phys Accel",
			"Virt Accel",
			"Vine Procs",
			"Vine Datas"
		};

		ID_OUT << "<div class=group>\n";
		for(type = 0 ; type < VINE_TYPE_COUNT ; type++)
		{
			list = vine_object_list_lock(&(vpipe->objs),(vine_object_type_e)type);
			ID_OUT << "<table>\n";
			ID_OUT << "<tr><th colspan=3>" << typestr[type] << "[" << list->length << "] </th><tr>\n";
			ID_OUT << "<tr><th>Address</th><th>Name</th><th>Type</th>\n";
			if(list->length)
			{
				utils_list_for_each(*list,itr)
				{
					obj = (vine_object_s*)itr->owner;
					switch(type)
					{
						case VINE_TYPE_PHYS_ACCEL:
							ID_OUT << "<tr><td>" << obj << "</td><td>" << obj->name << "</td><td>" << vine_accel_type_to_str(((vine_accel_s*)obj)->type) << "</td><tr>\n";
							break;
						case VINE_TYPE_VIRT_ACCEL:
							ID_OUT << "<tr><td>" << obj << "</td><td>" << obj->name << "</td><td>" << vine_accel_type_to_str(((vine_vaccel_s*)obj)->type) << "</td><tr>\n";
							break;
						case VINE_TYPE_PROC:
							ID_OUT << "<tr><td>" << obj << "</td><td>" << obj->name << "</td><td>" << vine_accel_type_to_str(((vine_proc_s*)obj)->type) << "</td><tr>\n";
							break;
						case VINE_TYPE_DATA:
							ID_OUT << "<tr><td>" << obj << "</td><td>" << "Data" << "</td><td>Size:" << ((vine_data_s*)obj)->size << "</td><tr>\n";
							break;
						default:
							ID_OUT << "<tr><td>" << obj << "</td><td>" << obj->name << "</td><td>Unknown</td><tr>\n";
							break;
					}
				}
			}
			else
			{
				ID_OUT << "<tr><td colspan=3> No " << typestr[type] << "</td></tr>\n";
			}
			ID_OUT << "</table>\n";
			vine_object_list_unlock(&(vpipe->objs),(vine_object_type_e)type);
		}
		ID_OUT << "</div>\n";
	}

	if(!args["nobreak"])
	{
		ID_OUT << "<h1>Breakdowns</h1>\n";
		#ifdef BREAKS_ENABLE
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
			}
		}
		vine_object_list_unlock(&(vpipe->objs),VINE_TYPE_PROC);
		#else
		ID_OUT << "Breakdowns are not enabled!\n";
		#endif
	}
	if(!args["embed"])
	{
		ID_OUT << "</body>\n";
		ID_OUT << "</html>\n";
	}
	out.flush();
}

WebUI :: WebUI(std::map<std::string,bool> & args)
: args(args)
{

}
