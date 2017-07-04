#include <vine_pipe.h>
#include <arch/alloc.h>
#include <core/vine_object.h>
#include <stdio.h>
#include "Poco/Net/HTTPServer.h"
#include "Poco/Util/ServerApplication.h"
#include "Poco/Net/HTTPRequestHandler.h"
#include <Poco/Net/HTTPResponse.h>
#include <Poco/Net/HTTPServerRequest.h>
#include <Poco/Net/HTTPServerResponse.h>
#include <Poco/URI.h>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <set>

using namespace Poco;
using namespace Poco::Util;
using namespace Poco::Net;

const char * units[] = {"b ","Kb","Mb","Gb","Tb",0};

vine_pipe_s *vpipe;

int id_lvl = 1;

#define ID_OUT {for(int cnt = 0 ; cnt <= id_lvl ; cnt++)out << '\t';}out << '\t'

std::vector<std::string> pallete;

const char * ns_to_secs[] = {"ns","us","ms","s"};

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

const char * normalize(const char * label,size_t size)
{
	static char buff[1024];
	snprintf(buff,sizeof(buff),"%*c<tr><th>%s</th><td>%s</td></tr>\n",id_lvl-1,'\t',label,autoRange(size,units,1024).c_str());
	return buff;
}

int bar_count = 0;

#ifdef BREAKS_ENABLE

void addBarSlice(std::ostream & os,int bar_count,std::string color,unsigned long long value)
{
	os << "<div id='slice"<<bar_count<< "_" << color <<"' class=slice1 style = 'flex-grow:" << value << ";background-color:#" << color << ";'></div>\n";
}

std::string generateBreakBar(std::ostream & out,utils_breakdown_stats_s * breakdown)
{
	int samples = breakdown->samples;
	std::ostringstream bar;
	std::ostringstream heads;
	std::ostringstream percs;
	std::ostringstream raw_vals;
	std::ostringstream total_bar;
	bar << "<div class=bar>\n";
	total_bar << "<div class=tot_bar>\n";
	heads << "<tr><th style='border: none;'></th><th class='btn1' onClick=\"toggleSlice(this,'slice" <<bar_count<< "_" << pallete[0] << "')\" style='background-color:#" << pallete[0] << "'>";

	char * s = breakdown->heads;
	int parts = 0;
	while(*s)
	{
		if(*s == ',')
		{
			heads << "</th><th class='btn1' onClick=\"toggleSlice(this,'slice" <<bar_count<< "_" << pallete[parts+1] << "')\" style='background-color:#" << pallete[parts+1] << "'>";
			parts++;
		}
		else
		{
			if(*s == '_')
				heads << ' ';
			else
				heads << *s;
		}
		s++;
	}

	addBarSlice(total_bar,bar_count,pallete[parts],breakdown->part[BREAKDOWN_PARTS]);
	total_bar << "</div>\n";

	if(breakdown->part[BREAKDOWN_PARTS])
	{
		heads << "Total</th><th class=invisible></th><th onClick=\"toggleSlice(this,'slice" <<bar_count<< "_" << pallete[parts+1] << "')\" style='background-color:#" << pallete[parts+1] << "'>Task Interval</th></tr>\n";
		heads << "<tr><th>Time</th>";
		percs << "<tr><th>Percent</th>";
		raw_vals << "<tr><th>Time(ns)</th>";
		for(int part = 0 ; part < parts ; part++)
		{
			float perc = (100.0*breakdown->part[part])/breakdown->part[BREAKDOWN_PARTS];
			heads << "<td>" << autoRange(breakdown->part[part]/(float)samples,ns_to_secs,1000) << "</td>";
			percs << "<td>" << ((int)(1000*perc))/1000.0 << " <div class=u>%</div></td>";
			addBarSlice(bar,bar_count,pallete[part],breakdown->part[part]);
			raw_vals << "<td>" << breakdown->part[part] << "<div class=u>ns</div></td>";
		}
		addBarSlice(bar,bar_count,pallete[parts+1],breakdown->part[BREAKDOWN_PARTS+1]);
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
class WebHandler : public HTTPRequestHandler
{
	virtual void handleRequest(HTTPServerRequest & request,HTTPServerResponse & response)
	{
		int type;
		utils_list_s *list;
		utils_list_node_s *itr;
		vine_object_s *obj;
		response.setStatus(HTTPResponse::HTTP_OK);
		response.setContentType("text/html");
		URI uri(request.getURI());

		std::ostream& out = response.send();

		ID_OUT <<
"<!DOCTYPE html>\n"
"	<html>\n"
"		<head>\n"
//"			<meta http-equiv=\"refresh\" content=\"1\">\n"
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


ID_OUT << "		</head>\n"
"		<body>\n";

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



		arch_alloc_stats_s stats = arch_alloc_stats(&(vpipe->allocator));
		ID_OUT << "<h1>Allocator status</h1>\n";
		ID_OUT << "<div class=group>\n";
		id_lvl++;
		ID_OUT << "<table>\n";
		id_lvl++;
		ID_OUT << "<tr><th colspan=2>All Partitions</th></tr>\n";
		ID_OUT <<normalize("Partitions",stats.mspaces);
		ID_OUT <<normalize("Space",stats.total_bytes);
		ID_OUT <<normalize("Used",stats.used_bytes);
		ID_OUT <<normalize("Free",stats.total_bytes-stats.used_bytes);
		#ifdef ALLOC_STATS
		ID_OUT <<normalize("Failed allocations",stats.allocs[0]);
		ID_OUT <<normalize("Good allocations",stats.allocs[1]);
		ID_OUT <<normalize("Frees",stats.frees);
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
				ID_OUT <<normalize("Space",stats.total_bytes);
				ID_OUT <<normalize("Used",stats.used_bytes);
				ID_OUT <<normalize("Free",stats.total_bytes-stats.used_bytes);
				ID_OUT << "</table>\n";
				id_lvl--;
			}
		}
		while(stats.mspaces);
		ID_OUT << "</div>\n";
		id_lvl--;

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
					ID_OUT << "<h2>" << vine_accel_type_to_str(proc->type) << "::" << obj->name << "(" << samples << " samples,task average):</h2>\n";
					ID_OUT << "<a href='reset?" << (void*)&(proc->breakdown) << "'>Reset breakdown</a>\n";
					out << generateBreakBar(out,&(proc->breakdown));
				}
			}
			vine_object_list_unlock(&(vpipe->objs),VINE_TYPE_PROC);
		#else
			ID_OUT << "Breakdowns are not enabled!\n";
		#endif

		ID_OUT << "</body>\n";
		ID_OUT << "</html>\n";
		out.flush();
	}
};

class WebHandlerFactory : public HTTPRequestHandlerFactory
{
	public:
		virtual HTTPRequestHandler* createRequestHandler(const HTTPServerRequest& rq)
		{
			return new WebHandler();
		}
};

class Server : public ServerApplication
{
	void initialize(Application & self)
	{
		server = new HTTPServer(new WebHandlerFactory(),8888,new HTTPServerParams());
	}

	int main(const std::vector < std::string > & args)
	{
		server->start();
		waitForTerminationRequest();
		server->stop();
	}

	private:
		HTTPServer *server;
};

int main(int argc,char * argv[])
{

	for(int r = 0 ; r < 16 ; r++)
		for(int g = 0 ; g < 16 ; g++)
			for(int b = 0 ; b < 16 ; b++)
			{
				std::string c = "";
				if(r < 10)
					c += '0'+r;
				else
					c += 'A'+(r-10);
				if(g < 10)
					c += '0'+g;
				else
					c += 'A'+(g-10);
				if(b < 10)
					c += '0'+b;
				else
					c += 'A'+(b-10);
				pallete.push_back(c);
			}
	std::shuffle(pallete.begin(),pallete.end(),std::default_random_engine(0));


	vpipe = vine_talk_init();
	if(!vpipe)
	{
		fprintf(stderr,"Could not get vine_pipe instance!\n");
		return -1;
	}

	Server app;
	app.run(argc,argv);

	vine_talk_exit();
	return 0;
}
