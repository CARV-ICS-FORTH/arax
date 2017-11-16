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
			void taskExecutionGraph(std::ostream & os);
			virtual Poco::Net::TCPServerConnection* createConnection(const Poco::Net::StreamSocket& sock);
		private:
			typedef struct JobTrace
			{
				JobTrace();
				uint64_t getStart() const;
				uint64_t getEnd() const;
				uint64_t getDuration() const;
				void histogram(std::ostream & os);
				void addSample(const Sample & sample);
				size_t getSize();
				const std::vector<Sample> & getSamples() const;
				static bool byStartTime(const std::pair<void* const, JobTrace*> & a,const std::pair<void* const, JobTrace*> & b);
				static bool byStartTimeP(const JobTrace* a,const JobTrace* b);
				private:
					std::vector<Sample> samples;
					uint64_t start;
					uint64_t end;
					std::mutex lock;
			}JobTrace;

			std::mutex map_lock;
			std::unordered_map<void*,JobTrace*> jobs;
	};
#endif
