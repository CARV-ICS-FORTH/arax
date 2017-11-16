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
			typedef struct Sample
			{
				int sample_id;
				utils_breakdown_instance_s stats;
				uint64_t getStart() const;
				uint64_t getEnd() const;
				uint64_t getDuration() const;
				uint64_t operator[](int part) const;
				static bool byStartTime(const Sample & a,const Sample & b);
				static bool byEndTime(const Sample & a,const Sample & b);
				static bool byDuration(const Sample & a,const Sample & b);
			}Sample;
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
			typedef struct JobTrace
			{
				uint64_t getStart();
				uint64_t getEnd();
				uint64_t getDuration();
				void histogram(std::ostream & os);
				void addSample(const Sample & sample);
				size_t getSize();
				const std::vector<Sample> & getSamples() const;
				private:
					std::vector<Sample> samples;
					std::mutex lock;
			}JobTrace;

			std::mutex map_lock;
			std::unordered_map<void*,JobTrace*> jobs;
	};
#endif
