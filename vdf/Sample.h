#ifndef VDF_SAMPLE_HEADER
	#define VDF_SAMPLE_HEADER
	#include <Poco/Net/StreamSocket.h>
	#include "utils/breakdown.h"

	class Sample
	{
		public:
			bool recv(Poco::Net::StreamSocket & sock);
			void setID(int id);
			int getID();
			void * getVAccel();
			void * getPAccel();
			static bool byStartTime(const Sample & a,const Sample & b);
			static bool byEndTime(const Sample & a,const Sample & b);
			static bool byDuration(const Sample & a,const Sample & b);
			std::string getPAccelDesc();
			std::string getTaskDesc();
			uint64_t getStart() const;
			uint64_t getEnd() const;
			uint64_t getDuration() const;
			uint64_t operator[](int part) const;
	private:
			int id;
			utils_breakdown_instance_s stats;
	};
#endif
