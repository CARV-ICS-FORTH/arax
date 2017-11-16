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
	Sample data;

	while(ss.receiveBytes(&data.stats,sizeof(data.stats)) > 0)
	{
		collector.map_lock.lock();
		auto ins = collector.jobs.emplace(data.stats.vaccel,job);
		collector.map_lock.unlock();

		if(!ins.second)
		{	// Already had one
			job = ins.first->second;
		}
		else
		{	// Used my spare, create new
			spare_trace = new JobTrace();
		}

		job->addSample(data);

		job = spare_trace;
	}
}
Collector :: Collector(uint16_t port)
: TCPServer(this,port)
{

}

void Collector :: JobTrace :: histogram(std::ostream & os)
{
	double sdx = 950.0/(samples.size()+2);
	double bar_width = 950.0/(samples.size()+3);
	double max_task_time = std::max_element(
		samples.begin(),samples.end(),Sample::byDuration)->getDuration();

		os << "<svg data-sort=\"time\" data-boff=" << sdx << " id=\"" << (void*)this << "\"class=\"chart\" width=\"1050\" height=\"650\">";
 	os << "<text x=25 y=40 font-size=30>Job Task Latency</text>";
	os << "<text id='title' onClick=\"resortGraph(this,['time','cdf'])\" x=975 text-anchor=\"end\" y=40 font-size=30>&#x1f441; Start Time</text>";
	os << "<text id='task_stuff' x=525 y=40 font-size=20 text-anchor='middle' ></text>";
	os << "<g transform=\"translate(25,50)\">\n";	// Graph area from 25,25 to 425,325

	std::sort(samples.begin(),samples.end(),Sample::byDuration);

	for(int divs = 0 ; divs <= 10 ; divs++)
	{
		double y = (float)(divs*575)/10;
		double time = (max_task_time/10.0)*divs;
		os << "<line stroke-width=1 stroke=\"black\" x1=0 x2=950 y1=" << y;
		os << " y2=" << y << "></line>\n";
		os << "<text y=" << 575-y;
		os << " x=950 font-size:20>";
		os << autoRange(time,ns_to_secs,1000,10) << "</text>\n";
	}
	int hist_id = 1;
	for( auto sample : samples)
	{
		double h = (sample.getDuration()/max_task_time)*575;
		os << "<rect onmouseover=barInfo(this,\"" << (void*)this << "\"," << samples.size() << ") x=" << sample.sample_id*sdx;
		os << " width=" << bar_width;
		os << " y=" << 575-h;
		os << " height=" << h;
		os << " time_id=" << sample.sample_id;
		os << " hist_id=" << hist_id;
		os << " duration=\"" << autoRange(sample.getDuration(),ns_to_secs,1000,10) << "\">";
		os << "</rect>";
		hist_id++;
	}
	os << "</g></svg>";
}

void Collector :: JobTrace :: addSample(const Sample & sample)
{
	lock.lock();
	samples.push_back(sample);
	samples.back().sample_id = samples.size();
	lock.unlock();
}

size_t Collector :: JobTrace :: getSize()
{
	return samples.size();
}

const std::vector<Collector :: Sample> & Collector :: JobTrace :: getSamples() const
{
	return samples;
}

void Collector :: rawDump(std::ostream & os)
{
	map_lock.lock();
	os << "Jobs:" << jobs.size() << std::endl;
	for( auto job : jobs)
	{
		os << "<table>";
		os << _TR(_TH("Samples")+_TD(std::to_string(job.second->getSize())));
		os << _TR(_TH("Start")+_TD(autoRange(job.second->getStart(),ns_to_secs,1000)));
		os << _TR(_TH("End")+_TD(autoRange(job.second->getEnd(),ns_to_secs,1000)));
		os << _TR(_TH("Duration")+_TD(autoRange(job.second->getDuration(),ns_to_secs,1000)));
		job.second->histogram(os);
	}
	map_lock.unlock();
}

Poco::Net::TCPServerConnection* Collector :: createConnection(const Poco::Net::StreamSocket& sock)
{
	return new Collector::CollectorConnection(sock,*this);
}

uint64_t Collector :: Sample :: getStart() const
{
	return stats.start;
}

uint64_t Collector :: Sample :: getEnd() const
{
	return getStart()+getDuration();
}
uint64_t Collector :: Sample :: getDuration() const
{
	return stats.part[BREAKDOWN_PARTS];
}

uint64_t Collector :: Sample :: operator[](int part) const
{
	return stats.part[part];
}

bool Collector :: Sample :: byStartTime(const Sample & a,const Sample & b)
{
	return a.getStart()< b.getStart();
}

bool Collector :: Sample :: byEndTime(const Sample & a,const Sample & b)
{
	return a.getEnd() < b.getEnd();
}

bool Collector :: Sample :: byDuration(const Sample & a,const Sample & b)
{
	return a.getDuration() < b.getDuration();
}

uint64_t Collector :: JobTrace :: getStart()
{
	auto ret = std::min_element(samples.begin(),samples.end(),Sample::byStartTime);
	return ret->getStart();
}
uint64_t Collector :: JobTrace :: getEnd()
{
	auto ret =  std::max_element(samples.begin(),samples.end(),Sample::byEndTime);
	return ret->getEnd();
}

uint64_t Collector :: JobTrace :: getDuration()
{
		return getEnd()-getStart();
}

Collector :: CollectorConnection :: CollectorConnection(const Poco::Net::StreamSocket& s,Collector & collector)
:Poco::Net::TCPServerConnection(s), collector(collector)
{}
