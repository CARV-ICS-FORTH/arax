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

	while(data.recv(ss))
	{
		collector.map_lock.lock();
		auto ins = collector.jobs.emplace(data.getVAccel(),job);
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

void generateExecutionBarTask(std::ostream & os,std::string fill,double x,double y, double width)
{
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
	os << "<svg preserveAspectRatio=\"none\" viewBox=\"0 0 1150 " << std::max(50+40*jobs.size(),(size_t)(50+20*8)) <<  "\" class='exec_chart' view=0 \">";

	for(int j = 0 ; j < jobs.size() ; j++)
		os << _TEXT("Job"+_S(j),"font-size=20 x=0 y="+_S(70+j*40));

	os << _TEXT("&#x1f441;",
			   "font-size=20 x=20 y=45 onClick=changeView(this)");

	std::map<std::string,int> accel_ids;

	os << "<g id='view0' transform=\"translate(75,25)\">\n";

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

	os << "</g><g class=hide id='view1' transform=\"translate(75,25)\">\n";

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

void Collector :: rawDump(std::ostream & os)
{
	map_lock.lock();

	if(jobs.size())
		taskExecutionGraph(os);
	else
		os << "No telemetry data!\n";

	map_lock.unlock();
}

Poco::Net::TCPServerConnection* Collector :: createConnection(const Poco::Net::StreamSocket& sock)
{
	return new Collector::CollectorConnection(sock,*this);
}

Collector :: CollectorConnection :: CollectorConnection(const Poco::Net::StreamSocket& s,Collector & collector)
:Poco::Net::TCPServerConnection(s), collector(collector)
{
}
