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
#include <unistd.h>
#include <map>
#include "Misc.h"
#include "WebUI.h"

using namespace Poco;
using namespace Poco::Util;
using namespace Poco::Net;


char hostname[1024];

vine_pipe_s *vpipe;

std::map<std::string,bool> args;

std::vector<std::string> pallete;

const char * ns_to_secs[] = {"ns","us","ms","s"};

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

class WebHandlerFactory : public HTTPRequestHandlerFactory
{
	public:
		virtual HTTPRequestHandler* createRequestHandler(const HTTPServerRequest& rq)
		{
			return new WebUI(args);
		}
};

class Server : public ServerApplication
{
	void initialize(Application & self)
	{
		std::cerr << "VDF at port " << port << std::endl;
		server = new HTTPServer(new WebHandlerFactory(),port,new HTTPServerParams());
	}

	int main(const std::vector < std::string > & args)
	{
		server->start();
		waitForTerminationRequest();
		server->stop();
	}

	HTTPServer *server;
	int port;

	public:
		Server(int port)
		:port(port)
		{}
};



int main(int argc,char * argv[])
{
	int port = 8888;

	for(int arg = 0 ; arg < argc ; arg++)
	{
		if(atoi(argv[arg]))
			port = atoi(argv[arg]);
		else
			args[argv[arg]] = true;
	}

	gethostname(hostname,1024);

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

	Server app(port);
	app.run(argc,argv);

	vine_talk_exit();
	return 0;
}
