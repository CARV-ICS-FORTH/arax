#ifndef VDF_WEB_UI_HEADER
	#define VDF_WEB_UI_HEADER
	#include <map>
	#include <Poco/Net/HTTPRequestHandler.h>
	#include <Poco/Net/HTTPServerRequest.h>
	#include <Poco/Net/HTTPResponse.h>
	#include <Poco/Net/HTTPServerResponse.h>

	class WebUI : public Poco::Net::HTTPRequestHandler
	{
		virtual void handleRequest(Poco::Net::HTTPServerRequest & request,Poco::Net::HTTPServerResponse & response);
		public:
			WebUI(std::map<std::string,bool> & args);
		private:
			std::map<std::string,bool> args;
	};
#endif
