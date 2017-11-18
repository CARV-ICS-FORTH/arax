#ifndef VDF_COLLECTOR_HEADER
	#define VDF_COLLECTOR_HEADER
	#include <Poco/Net/TCPServerConnection.h>
	#include <Poco/Net/StreamSocket.h>
	#include "Poco/Net/TCPServer.h"
	#include <unordered_map>
	#include <mutex>
	#include <ostream>
	#include "Sample.h"
	#include "JobTrace.h"

	class Collector : public Poco::Net::TCPServer, public Poco::Net::TCPServerConnectionFactory
	{
		public:
			class CollectorConnection : public Poco::Net::TCPServerConnection
			{
				virtual void run();
				public:
					CollectorConnection(const Poco::Net::StreamSocket& s,Collector & collector);
				private:
					Collector & collector;
			};
		private:

			uint64_t start_time;
			std::mutex map_lock;
			std::unordered_map<void*,JobTrace*> jobs;
		public:
			Collector(uint16_t port);
			void rawDump(std::ostream & os);
			static void generateTaskExecutionGraph(std::ostream & os,const std::vector<JobTrace*> & jobs);
			void taskExecutionGraph(std::ostream & os);
			virtual Poco::Net::TCPServerConnection* createConnection(const Poco::Net::StreamSocket& sock);
	};
#endif
