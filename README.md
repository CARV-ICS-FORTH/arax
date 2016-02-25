This library aims to implement the main communication layer between the
Application VMs and the Appliance VMs.

# Design

![IMAGE](raw/master/ high_level.svg)

As discussed in 24-02-2016 with mavridis, manospavlidakis, nchrisos, and
zakkak, we decided the following steps:

1. Implement a single work-queue per Appliance VM.  This queue must be a
   multi-producer/single-consumer concurrent queue (preferably lock
   free).  
   
   Such a design significantly simplifies the code complexity in the
   Appliance VM side, but might as well hinter performance.

2. If a single queue appears to hinter performance by becoming a
   bottleneck, we decided to experiment with `N` work-queues, where `N`
   is the number of physical cores available in the server.  Then, each
   VM can only use one out of the `N` queues based on its ID (we will
   probably use a simple hash function here).

## Concerns

* Interruption/event driven VS polling
* Atomics VS locks
