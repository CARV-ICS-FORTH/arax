![VineTalk Logo](docs/logo.png)

This library aims to implement the main communication layer between the
Application VMs and the Appliance VMs.

# Requirements

To build Vinetalk you will require cmake, ccmake and (optionaly but recomened) libpoco and doxygen:

## ArchLinux

``sudo pacman -S cmake poco doxygen``

## CentOS

``sudo yum install cmake poco-foundation poco-net doxygen``

## Ubuntu

``sudo apt-get install cmake cmake-curses-gui libpoco-dev doxygen``

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

# API Documentation

To generate documentation see the `Build doxygen documentation` section below.

# Building

Vinetalk is built as a shared library(libvine.so), using cmake and make:

First build and navigate to your build folder:

<code>mkdir build;cd build</code>

You must then configure your build using ccmake or cmake directly:

## Configure with CCMake

Run <code>ccmake ..</code> in your build folder and press `c` once:

![ccmake screenshot](docs/ccmake_scr.png)

Every line correspond to a build option(see below for option descriptions).
To change/set an option press enter, this will toggle a Boolean flag or allow you to edit a string option.
For string press enter again to end string input.

Once you have configured your build, press `c` followed by `g`.

## Configure with CMake

To configure using, on the build folder you type:

<code>cmake [Configuration Options] ..</code>


### Configuration Options

| Option                                           | Description                                                                          |
|--------------------------------------------------|--------------------------------------------------------------------------------------|
|-DALLOC_STATS=ON&#124;OFF                         | Enable Allocator Statistics                                                          |
|-DARCH_ALLOC_MAX_SPACE=NUMBER                     | Set maximum usable allocator space                                                   |
|-DBREAKS_ENABLE                                   | Enable breakdown reporting                                                           |
|-DBUILD_TESTS=ON&#124;OFF                         | Build unit tests                                                                     |
|-DCMAKE_BUILD_TYPE=Debug                          | Produce debug symbols                                                                |
|-DCOVERAGE=ON&#124;OFF                            | Enable gcov coverage                                                                 |
|-DJVineTalk                                       | Build java Vinetalk wrappers                                                         |
|-DTRACE_ENABLE=ON&#124;OFF                        | Enable trace file creation                                                           |
|-DMMAP_POPULATE=ON&#124;OFF                       | Add MAP_POPULTE flag in mmap. This will make vine_talk_init() slower, use wisely.    |
|-DUTILS_QUEUE_CAPACITY=NUMBER                     | Maximum number of outstanding tasks per task queue (Up to 65536), MUST BE power of 2 |
|-DVINE_CONFIG_FILE=FILE                           | Vinetalk configuration file                                                          |
|-Dasync_architecture=spin&#124;mutex&#124;ivshmem | Method used to ensure ordering                                                       |
|-Dtarget_architecture=shm                         | Method used to transfer data                                                         |

## CCMake

Run <code>ccmake ..</code> in your build forder and press c:

## Build with Make

After configuring, run <code>make</code>

### Build doxygen documentation

After configuring, run <code>make doc</code>

## Testing

After building with tests enabled, you can run tests with <code>make test</code>.

## Install

VineTalk can be 'installed' in two ways.
System Wide install is the recomened method if deploying on a dedicated machine and have root/sudo access.
User Specific installation is recomended if deploying in a shared machine with multiple users and dont have root/sudo access.

### System Wide Install

After a successful build, run <code>make install</code>, with root privileges.

### User Specific Install

You can use the LD_LIBRARY_PATH eviroment variable to load VineTalk from the build path.

<code>export LD_LIBRARY_PATH=<VineTalk build path></code>

To find the apropriate VineTalk build path, run:

<code>make VineTalkBuildPath</code>

## Using the Vine Talk Library

After a successful build your build directory will have a libvine.so file as well as
an include folder. Add your build path as a library path and link with vinetalk <code>-lvine</code>.
Also add the build/includes folder to your gcc include paths <code>-Ibuild/includes</code>.

# Configuration

In order to configure the vine_pipe endpoints, the user must provide
architecture specific options.

These configuration options are stored at ~/.vinetalk and follow the format
specified in utils/config.h.

The sections bellow specify the required keys for each supported vinetalk
architecture:

## Breakdowns

The BREAKS_ENABLE allow the generation of breakdowns(.brk) and headers (.hdr) for performance evaluation
of the VineTalk system.

Steps:
- Enable BREAKS_ENABLE option and rebuild VineTalk.
- Run applications...
- When the controller terminates, you will have .hdr and .brk files for all procedures that run.

## Vdf

Vdf is a program located at the examples folder, allowing run time inspection of allocator statistics and breakdowns(and more to come).
To be built it requires the Poco framework to be installed.

After running it spawns a web server at localhost:8888.
The web ui allows inspection of allocator statistics and breakdowns.

## shm

Shm implements the vinetalk API/protocol over a shared segment
(POSIX or ivshmem).

### Required Configuration Keys

| Option   | Description                                      |
|----------|--------------------------------------------------|
| shm_file | A file path specifying the shared segments file. |
| shm_size | The size of the shared segment in bytes.         |

### Optional Configuration Keys

| Option      | Description                                                                                                             |
|-------------|-------------------------------------------------------------------------------------------------------------------------|
| shm_trunc   | A boolean (0,1) setting if the shm_file should be truncated during initialization.                                      |
| shm_off     | Start mmap from the given byte offset instead from 0.Can be used to split a single shm to multiple vine_pipe instances. |
| shm_ivshmem | Boolean , set to 1 if running inside a Vm with ivshmem.                                                                 |

## tracer

Tracer implements an api that tracing vine_talk interface.

### Required Configuration Keys

### Optional Configuration Keys

| Option             | Description                                                           |
|--------------------|-----------------------------------------------------------------------|
| tracer_buffer_size | The size of log buffer in bytes,default is 100 entries                |
| tracer_path        | Existing folder, where trace log files will be placed, default is cwd |

## VDF

VDF is a monitoring tool for Vinetalk, exposing statistics through a web interface.
It accepts the following arguements:

| Arguement          | Description                                                           |
|--------------------|-----------------------------------------------------------------------|
| embed              | Skips html and head tags from output, allowing output to be embeded   |
| noalloc            | Dont show allocation statistics                                       |
| noobj              | Dont show object statistics                                           |
| nobreak            | Dont show breakdowns                                                  |
