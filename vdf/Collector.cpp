#include "Collector.h"
#include <iostream>
#include <string>
#include "Misc.h"
#include <algorithm>
#include <map>

extern std::vector<std::string> pallete;

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
	struct timespec now;
	clock_gettime(CLOCK_REALTIME,&now);
	start_time = now.tv_sec*1000000000+now.tv_nsec;

}

void Collector :: JobTrace :: histogram(std::ostream & os,float ratio)
{
	double sdx = 950.0/(samples.size()+2);
	double bar_width = 950.0/(samples.size()*1.5);
	double max_task_time = std::max_element(
		samples.begin(),samples.end(),Sample::byDuration)->getDuration();

		os << "<svg style=\"display:flex;flex:" << ratio << "\" viewBox=\"0 0 1050 650\" data-sort=\"time\" data-boff=" << sdx << " id=\"" << _S((uint64_t)this) << "\"class=\"bar_chart\" width=\"1050\" height=\"650\">";
	os << "<text x=25 y=40 font-size=30>Job Task Latency</text>";
	os << "<text id='title' onClick=\"resortGraph(this,['time','cdf'])\" x=975 text-anchor=\"end\" y=40 font-size=30>&#x1f441; Start</text>";
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
		os << _RECT("",sample.sample_id*sdx,575-h,bar_width,h,
					" time_id=" + _S(sample.sample_id) + " hist_id=" + _S(hist_id) +
					" onmouseover=barInfo(this,\"" + _S((uint64_t)this) + "\"," + _S(samples.size())+")"
		);
		hist_id++;
	}
	os << "</g></svg>";
}

void generateExecutionBarTask(std::ostream & os,std::string fill,double x,double y, double width)
{
//	os << "<rect fill=\"#" << fill << "4\" x=\"" << x << "\" y=\"" << y
//	<< "\" height=\"30\" width=\"" << width << "\"></rect>" << std::endl;

	os << _RECT(fill+"4",x,y,width,30,"");
}

void generateColorTextBox(std::ostream & os,double x,double y,double width,std::string text,std::string fill)
{
	os <<
		tag_gen("g",
			_RECT(fill+"4",0,0,width,20,"")+
			_TEXT(text,"y=20 text-anchor=middle fill='black' x="+_S(width/2)),
		  "transform=\'translate("+_S(x)+","+_S(y)+")\'"
		);
}

void Collector :: generateTaskExecutionGraph(std::ostream & os,const std::vector<JobTrace*> & jobs)
{
	os << "<svg preserveAspectRatio=\"none\" viewBox=\"0 0 1150 " << std::max(50+40*jobs.size(),(size_t)(50+20*8)) <<  "\" class='exec_chart' data-view='jobs' \">";

	for(int j = 0 ; j < jobs.size() ; j++)
		os << _TEXT("Job"+_S(j),"font-size=20 x=0 y="+_S(70+j*40));

	os << _TEXT("&#x1f441;",
			   "font-size=20 x=20 y=45 onClick=changeView(this,['jobs','accels'])");

	std::map<std::string,int> accel_ids;

	os << "<g data_view='jobs' transform=\"translate(75,25)\">\n";

	int task_uid = 0;
	uint64_t start = jobs.front()->getStart();
	uint64_t end = jobs.back()->getEnd();
	double duration = end-start;
	int jy = 25;


	for(auto job : jobs)
	{
		for(auto sample : job->getSamples())
		{
			if(!accel_ids.count(sample.getPAccelDesc()))
				accel_ids[sample.getPAccelDesc()] = accel_ids.size();
			generateExecutionBarTask(
				os,
				pallete[accel_ids[sample.getPAccelDesc()]],
				((sample.getStart()-start)*1000)/duration,
				jy,
				(sample.getDuration()*1000)/duration
			);
		}
		jy += 40;
	}

	int x = 0;
	double dx = 1000.0/accel_ids.size();

	for(auto accel : accel_ids)
	{
		generateColorTextBox(os,x,0,dx,accel.first,pallete[accel_ids[accel.first]]);
		x += dx;
	}

	os << "</g><g class=hide data-view='accels' transform=\"translate(75,25)\">\n";

	for(auto accel : accel_ids)
	{
		generateColorTextBox(os,0,accel.second*20,100,accel.first,pallete[accel_ids[accel.first]]);
	}

	os << "</g></svg>";
}

void Collector :: taskExecutionGraph(std::ostream & os)
{
	std::vector<JobTrace *> sorted_jobs;

	for (const auto &s : this->jobs)
		sorted_jobs.push_back(s.second);

	std::sort(sorted_jobs.begin(),sorted_jobs.end(),JobTrace::byStartTimeP);




	std::vector<std::vector<JobTrace*>> sample_groups;

	uint64_t last_end = 0;
	for(auto job : sorted_jobs)
	{
		if(job->getStart() > last_end)
		{
			sample_groups.resize(sample_groups.size()+1);
		}
		last_end = job->getEnd();
		sample_groups.back().push_back(job);
	}

	for(auto group : sample_groups)
	{
		os << "<div class=vgroup>\n";
		os << "<h3> " << group.size() << " jobs between " << autoRange(group.front()->getStart()-start_time,ns_to_secs,1000) << " - "<< autoRange(group.back()->getEnd()-start_time,ns_to_secs,1000) << "</h3>";
		os << "<div class=hgroup>\n";
		for(auto job : group)
		{
			job->histogram(os,1.0/group.size());
		}
		os << "</div>\n";
		generateTaskExecutionGraph(os,group);
		os << "</div>\n";
	}

}

void Collector :: JobTrace :: addSample(const Sample & sample)
{
	lock.lock();
	samples.push_back(sample);
	if(sample.getStart() < start)
		start = sample.getStart();
	if(sample.getEnd() > end)
		end = sample.getEnd();
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

bool Collector :: JobTrace :: byStartTime(const std::pair<void* const, Collector::JobTrace*> & a,const std::pair<void* const, Collector::JobTrace*> & b)
{
	return a.second->getStart() < b.second->getStart();
}

bool Collector :: JobTrace :: byStartTimeP(const JobTrace* a,const JobTrace* b)
{
		return a->getStart() < b->getStart();
}

void Collector :: rawDump(std::ostream & os)
{
	map_lock.lock();

	taskExecutionGraph(os);

	map_lock.unlock();
}

Poco::Net::TCPServerConnection* Collector :: createConnection(const Poco::Net::StreamSocket& sock)
{
	return new Collector::CollectorConnection(sock,*this);
}

std::string Collector :: Sample :: getPAccelDesc()
{
	if(!stats.paccel)
		return "Unknown";
	return ((vine_object_s*)stats.paccel)->name;
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

Collector :: JobTrace :: JobTrace()
: start(-1)
{

}

uint64_t Collector :: JobTrace :: getStart() const
{
	return start;
}
uint64_t Collector :: JobTrace :: getEnd() const
{
	return end;
}

uint64_t Collector :: JobTrace :: getDuration() const
{
		return getEnd()-getStart();
}

Collector :: CollectorConnection :: CollectorConnection(const Poco::Net::StreamSocket& s,Collector & collector)
:Poco::Net::TCPServerConnection(s), collector(collector)
{
}
