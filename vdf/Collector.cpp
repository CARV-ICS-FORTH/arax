#include "Collector.h"
#include <iostream>
#include <string>
#include "Misc.h"
#include <algorithm>

void Collector :: CollectorConnection :: run()
{
	JobTrace * job = new JobTrace();
	JobTrace * spare_trace = job;
	Poco::Net::StreamSocket& ss = socket();
	utils_breakdown_instance_s data;

	while(ss.receiveBytes(&data,sizeof(data)) > 0)
	{
		collector.map_lock.lock();
		auto ins = collector.jobs.emplace(data.vaccel,job);
		collector.map_lock.unlock();

		if(!ins.second)
		{	// Already had one
			job = ins.first->second;
		}
		else
		{	// Used my spare, create new
			spare_trace = new JobTrace();
		}

		job->lock.lock();
		job->samples.push_back(data);
		job->lock.unlock();

		job = spare_trace;
	}
}
Collector :: Collector(uint16_t port)
: TCPServer(this,port)
{

}

void Collector :: rawDump(std::ostream & os)
{
	map_lock.lock();
	os << "Jobs:" << jobs.size() << std::endl;
	for( auto job : jobs)
	{
		os << "Job " << job.first << " Samples:" << job.second->samples.size() << " Start: " << job.second->getStart() << " End:" << job.second->getEnd() << " Duration:" << job.second->getDuration() << std::endl;
		os << "<table><tr><th>#</th></tr>";
		int tid = 0;
		for(auto sample : job.second->samples)
		{
			os << "<tr><td>" << tid++ << "</tid>";
			for(int c = -1 ; c < BREAKDOWN_PARTS+1 ; c++)
				os << _TD(std::to_string(sample.part[c])) << std::endl;
			os << "</tr>";
		}
		os << "</table>";
	}
	map_lock.unlock();
}

Poco::Net::TCPServerConnection* Collector :: createConnection(const Poco::Net::StreamSocket& sock)
{
	return new Collector::CollectorConnection(sock,*this);
}

bool min_start_time(const utils_breakdown_instance_s & a,const utils_breakdown_instance_s & b)
{
	return a.start < b.start;
}

bool max_end_time(const utils_breakdown_instance_s & a,const utils_breakdown_instance_s & b)
{
	return a.start+a.part[BREAKDOWN_PARTS] > b.start+b.part[BREAKDOWN_PARTS];
}

uint64_t Collector :: JobTrace :: getStart()
{
	auto ret = std::min_element(samples.begin(),samples.end(),min_start_time);
	return ret->start;
}
uint64_t Collector :: JobTrace :: getEnd()
{
	auto ret =  std::max_element(samples.begin(),samples.end(),max_end_time);
	return ret->start + ret->part[BREAKDOWN_PARTS];
}

uint64_t Collector :: JobTrace :: getDuration()
{
		return getEnd()-getStart();
}

Collector :: CollectorConnection :: CollectorConnection(const Poco::Net::StreamSocket& s,Collector & collector)
:Poco::Net::TCPServerConnection(s), collector(collector)
{}
