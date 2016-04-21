This library aims to implement the main communication layer between the
Application VMs and the Appliance VMs.

# Folder layout

* docs - Documentation
* examples - Usage examples of the code
* 3rdparty - Third-party libraries.
* include - Header files that expose the public interface
* src - Source code
    * core - The core implementation of the program/library
    * arch - The architectural specific implementations
    * utils: Contains helper modules, such as data structures, wrappers
      to external libraries, etc.
* tests - Contain the tests that should be run with `make test`

# Building

To build `libvine.a`.

``` shell
mkdir build
cd build
cmake ../
make libvine
```

To build `libvine.a` with debuging symbols:
``` shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ../
make libvine
```
# Configuration

In order to configure the vine_pipe endpoints, the user must provide
architecture specific options.

These configuration options are stored at ~/.vinetalk and follow the format
specified in utils/config.h.

The sections bellow specify the required keys for each supported vinetalk
architecture:

## shm

Shm implements the vinetalk API/protocol over a shared segment
(POSIX or ivshmem).

The required keys are the following:

shm_file: A file path specifying the shared segments file.

shm_size: The size of the shared segment in bytes.

Optional keys:

shm_trunc: A boolean (0,1) setting if the shm_file should be truncated
durring initialization.
## tracer

<Add stuff here>

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
