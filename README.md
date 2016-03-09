This library aims to implement the main communication layer between the
Application VMs and the Appliance VMs.

# Folder layout
*docs: Documentation
*examples: Examples utilizing the message-queue-lib
*empty: Empty library implementation, use it as a starting point.
*local_profile: Profiling library implementation, with local execution.

# Building

To build all library versions just make in the root directory.

# Adding a new library version

To add a new library version copy the empty library to a new folder(e.g. new_lib).
You can modify the vine_talk.c and add any files in the new directory.

And add the new version library (e.g. libvine_talk_new_lib.a) in the vine_talk_versions list in Makefile.

# Design

[Graphical representation](docs/high_level.svg) by mavridis.

## 26-02-2016

As discussed on 26-02-2016 with bilas, mavridis, manospavlidakis,
nchrisos, and zakkak, we decided to implement a single work-queue per
application thread.  This queue must be single-producer/single-consumer.

Such a design significantly simplifies the code complexity in the
Application VM side, but might as well hinter the Appliance VM's
performance if the number of queues starts getting really high.

To improve performance, on the Appliance VM side we will adopt the
mechanisms used in the sockets' implementation where a server may serve
a big number of sockets efficiently.

## 24-02-2016

As discussed on 24-02-2016 with mavridis, manospavlidakis, nchrisos, and
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
