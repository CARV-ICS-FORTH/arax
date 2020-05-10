#include "JobTrace.h"
#include "Misc.h"

void JobTrace :: histogram(std::ostream & os,float ratio)
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
			os << _RECT("",sample.getID()*sdx,575-h,bar_width,h,
						" time_id=" + _S(sample.getID()) +
						" hist_id=" + _S(hist_id) +
						" duration='" + autoRange(sample.getDuration(),ns_to_secs,1000,100) + "'" +
						" onmouseover=barInfo(this,\"" + _S((uint64_t)this) + "\"," + _S(samples.size())+")"
			);
			hist_id++;
		}
		os << "</g></svg>";
}

void JobTrace :: addSample(const Sample & sample)
{
	lock.lock();
	samples.push_back(sample);
	if(sample.getStart() < start)
		start = sample.getStart();
	if(sample.getEnd() > end)
		end = sample.getEnd();
	samples.back().setID(samples.size());
	lock.unlock();
}

size_t JobTrace :: getSize()
{
	return samples.size();
}

const std::vector<Sample> & JobTrace :: getSamples() const
{
	return samples;
}

bool JobTrace :: byStartTime(const std::pair<void* const, JobTrace*> & a,const std::pair<void* const, JobTrace*> & b)
{
	return a.second->getStart() < b.second->getStart();
}

bool JobTrace :: byStartTimeP(const JobTrace* a,const JobTrace* b)
{
	return a->getStart() < b->getStart();
}

JobTrace :: JobTrace()
: start(-1)
{

}

uint64_t JobTrace :: getStart() const
{
	return start;
}
uint64_t JobTrace :: getEnd() const
{
	return end;
}

uint64_t JobTrace :: getDuration() const
{
	return getEnd()-getStart();
}

