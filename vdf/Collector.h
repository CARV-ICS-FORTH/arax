#ifndef VDF_COLLECTOR_HEADER
	#define VDF_COLLECTOR_HEADER
	#include <Poco/Net/TCPServerConnection.h>
	#include <Poco/Net/StreamSocket.h>
	#include "Poco/Net/TCPServer.h"
	#include "utils/breakdown.h"
	#include <unordered_map>
	#include <mutex>
	#include <ostream>

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
			Collector(uint16_t port);
			void rawDump(std::ostream & os);
			virtual Poco::Net::TCPServerConnection* createConnection(const Poco::Net::StreamSocket& sock);
		private:
			typedef struct
			{
				std::vector<utils_breakdown_instance_s> samples;
				std::mutex lock;
				uint64_t getStart();
				uint64_t getEnd();
				uint64_t getDuration();
			}JobTrace;

			std::mutex map_lock;
			std::unordered_map<void*,JobTrace*> jobs;
	};
#endif
