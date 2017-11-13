#include "Collector.h"
#include <iostream>

void Collector :: CollectorConnection :: run()
{
	JobTrace * job = new JobTrace();
	Poco::Net::StreamSocket& ss = socket();
	utils_breakdown_instance_s data;

	ss.receiveBytes(&data,sizeof(data));

	collector.map_lock.lock();
	auto ins = collector.jobs.emplace(data.accel,job);
	collector.map_lock.unlock();

	if(!ins.second)
	{	// Already had one
		delete job;
		job = ins.first->second;
	}

	job->lock.lock();
	job->samples.push_back(data);
	job->lock.unlock();
}
Collector :: Collector(uint16_t port)
: TCPServer(this,port)
{

}

Poco::Net::TCPServerConnection* Collector :: createConnection(const Poco::Net::StreamSocket& sock)
{
	return new Collector::CollectorConnection(sock,*this);
}

Collector :: CollectorConnection :: CollectorConnection(const Poco::Net::StreamSocket& s,Collector & collector)
:Poco::Net::TCPServerConnection(s), collector(collector)
{}
