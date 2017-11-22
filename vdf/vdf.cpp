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
#include <set>
#include <unistd.h>
#include <map>
#include "Misc.h"
#include "WebUI.h"
#include "Collector.h"

using namespace Poco;
using namespace Poco::Util;
using namespace Poco::Net;


vine_pipe_s *vpipe;

std::map<std::string,bool> args;

class WebUIFactory : public HTTPRequestHandlerFactory
{
public:
	WebUIFactory(Collector* collector)
	: collector(collector)
	{

	}
	virtual HTTPRequestHandler* createRequestHandler(const HTTPServerRequest& rq)
	{
		return new WebUI(args,collector);
	}
private:
	Collector * collector;
};

class Server : public ServerApplication
{
	void initialize(Application & self)
	{
		std::cerr << "VDF at port " << port << std::endl;
		collector = new Collector(port+1);
		webui = new HTTPServer(new WebUIFactory(collector),port,new HTTPServerParams());

	}

	int main(const std::vector < std::string > & args)
	{
		collector->start();

		vpipe = vine_talk_init();
		if(!vpipe)
		{
			fprintf(stderr,"Could not get vine_pipe instance!\n");
			return -1;
		}
		webui->start();


		waitForTerminationRequest();
		collector->stop();
		webui->stop();
	}

	HTTPServer *webui;
	Collector *collector;
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

	Server app(port);
	app.run(argc,argv);

	vine_talk_exit();
	return 0;
}
