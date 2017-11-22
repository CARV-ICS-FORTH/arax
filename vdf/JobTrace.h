#ifndef VDF_JOBTRACE_HEADER
	#define VDF_JOBTRACE_HEADER
	#include <string>
	#include <vector>
	#include <string>
	#include <mutex>
	#include <ostream>
	#include "Sample.h"

	class JobTrace
	{
		public:
			JobTrace();
			uint64_t getStart() const;
			uint64_t getEnd() const;
			uint64_t getDuration() const;
			void histogram(std::ostream & os,float ratio);
			void addSample(const Sample & sample);
			size_t getSize();
			const std::vector<Sample> & getSamples() const;
			static bool byStartTime(const std::pair<void* const, JobTrace*> & a,const std::pair<void* const, JobTrace*> & b);
			static bool byStartTimeP(const JobTrace* a,const JobTrace* b);
			std::string getName();
		private:
			std::string name;
			std::vector<Sample> samples;
			uint64_t start;
			uint64_t end;
			std::mutex lock;
	};

#endif
