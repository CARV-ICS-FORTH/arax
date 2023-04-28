#include "AraxLibMgr.h"
#include "Formater.h"
#include "FreeThread.h"
#include "Scheduler.h"
#include "Services.h"
#include "VacBalancerThread.h"
#include "accelThread.h"
#include "arax_pipe.h"
#include "definesEnable.h"
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <future>

using namespace ::std;

arax_pipe_s *vpipe; /* get a pointer to the arax pipe */
void *shm = 0;      /* get a pointer to the share memory segment */
std::promise<void> shouldExit;

/* Define the function to be called when ctrl-c (SIGINT) signal is sent to
 * process*/
void signal_callback_handler(int signum)
{
    #if (DEBUG_ENABLED)
    cout << "Caught signal " << signum << endl;
    #endif
    shouldExit.set_value();
}

/*Deregister accelerators at termination*/
void deRegisterAccel(arax_pipe_s *vpipe_s,
  vector<AccelConfig *>           &accelSystemList)
{
    cout << "Deregister accelerators" << endl;
    vector<AccelConfig *>::iterator it;

    for (it = accelSystemList.begin(); it != accelSystemList.end(); ++it) {
        // add here accel size check
        auto total_size     = arax_accel_get_total_size((*it)->arax_accel);
        auto available_size = arax_accel_get_available_size((*it)->arax_accel);
        if (total_size != available_size) {
            /*throw std::runtime_error(std::string(__func__) + "phys accel Leak (" +
             *      std::to_string(total_size - available_size)
             + ") missing bytes");*/
            std::cerr << __func__ << " phys accel Leak ( "
                      << total_size - available_size << " ) missing bytes"
                      << std::endl;
        }
        arax_accel_release((arax_accel **) &((*it)->arax_accel));
    }
}

/*Creates one thread per physical accelerator*/
bool spawnThreads(arax_pipe_s *vpipe_s,
  vector<AccelConfig *>        &accelSystemList)
{
    /*iterate to the list with the accels from config file*/
    vector<AccelConfig *>::iterator itr;
    string msg;
    accelThread *thread;

    for (itr = accelSystemList.begin(); itr != accelSystemList.end(); ++itr) {
        // Create thread
        thread = threadFactory.constructType((*itr)->type_str, vpipe_s, *(*itr));
        if (!thread) {
            msg = "Could not create thread";
            goto FAIL;
        }
        (*itr)->accelthread = thread;
    }

    for (itr = accelSystemList.begin(); itr != accelSystemList.end(); ++itr) {
        (*itr)->accelthread->start();
        while ((*itr)->accelthread->isReadyToServe() == false)
            ;
    }

    return true;

FAIL:
    cerr << "While spawning: \'" << *itr << "\'\n"
         << msg << " for " << (*itr)->type_str << " device" << endl;
    return false;
}

/*Deregister threads */
void deleteThreads(vector<AccelConfig *> &accelSystemList)
{
    /*iterate to the list with the accels from config file*/
    vector<AccelConfig *>::iterator itr;

    for (itr = accelSystemList.begin(); itr != accelSystemList.end(); ++itr) {
        /*creates one thread per line from config file*/
        (*itr)->accelthread->terminate();
    }
    for (itr = accelSystemList.begin(); itr != accelSystemList.end(); ++itr) {
        (*itr)->accelthread->join();
        delete (*itr)->accelthread;
    }
}

arax_pipe_s *vpipe_s;

/*Main function*/
int main(int argc, char *argv[])
{
    /* get the directories that .so exist*/
    if (argc < 2) {
        cerr << "Usage:\n\t" << argv[0] << " config_file" << endl;
        return -1;
    }

    /*Create a config instance*/
    try {
        /*Config gets as arguments a file with the appropriate info*/
        Config config(argv[1]);

        /*Vector with the paths that libraries are located*/
        vector<string> paths = config.getPaths();

        paths.push_back(BUILTINS_PATH);

        cerr << "Library paths: " << Formater<string>("\n\t", "\n\t", "\n", paths)
             << endl;

        signal(SIGINT, signal_callback_handler);

        if (config.shouldCleanShm()) {
            if (arax_clean())
                std::cerr << "Cleaned up stale shm file!\n";
        }

        /*get the share mem segment*/
        vpipe_s = arax_controller_init_start();

        /* Load folder with .so*/
        AraxLibMgr *araxLibMgr = new AraxLibMgr();
        AraxLibMgr::KernelMap km;
        for (vector<string>::iterator itr = paths.begin(); itr != paths.end();
          itr++)
        {
            if (!araxLibMgr->loadFolder(*itr, km))
                return -1;
        }
        std::cout.width(20);
        std::cout << "-= Kernel Map =-" << std::endl << km << std::endl;

        araxLibMgr->startRunTimeLoader();

        cerr << "Supported threads: "
             << Formater<string>("\n\t", "\n\t", "\n", threadFactory.getTypes())
             << endl;
        cerr << "Supported Schedulers: "
             << Formater<string>("\n\t", "\n\t", "\n", schedulerFactory.getTypes())
             << endl;
        cerr << "Supported Services: "
             << Formater<string>("\n\t", "\n\t", "\n", serviceProvider.getTypes())
             << endl;

        cerr << config << endl;

        /*create an arax task pointer*/
        vector<AccelConfig *> vectorOfAccels;
        vectorOfAccels = config.getAccelerators();
        /*create threads*/
        if (!spawnThreads(vpipe_s, vectorOfAccels)) {
            cerr << "Failed to spawn threads, exiting" << endl;
            return -1;
        }

        new VacBalancerThread(vpipe_s, config);

        #ifdef FREE_THREAD
        FreeThread *f_th = new FreeThread(vpipe_s, config);
        #endif

        arax_controller_init_done();

        shouldExit.get_future().wait();
        cout << "Ctr+c is pressed EXIT!" << endl;

        /*keep the number of used accelerators*/
        int numOfUsedAccelerators = 0;
        /*total accels*/
        int numOfTotalAccelerators = 0;
        /*total tasks*/
        int totalTasks = 0;

        for (auto accel : vectorOfAccels) {
            std::cerr << accel->name << ", " << accel->accelthread->getServedTasks()
                      << " tasks" << std::endl;

            if (accel->accelthread->getServedTasks())
                numOfUsedAccelerators++;

            totalTasks += accel->accelthread->getServedTasks();
            numOfTotalAccelerators++;
        }

        std::cerr << "total, " << totalTasks << " tasks" << std::endl;
        std::cerr << "used accelerators, " << numOfUsedAccelerators << "/"
                  << numOfTotalAccelerators << std::endl;
        #ifdef FREE_THREAD
        f_th->terminate();
        #endif
        deleteThreads(vectorOfAccels);
        deRegisterAccel(vpipe_s, vectorOfAccels);
        std::cout << "Accelelators released" << std::endl;
        /*Delete libraries*/
        araxLibMgr->unloadLibraries(vpipe_s);
        std::cout << "Libraries unloaded" << std::endl;
        std::cout << "Libraries unloaded" << std::endl;
        delete araxLibMgr;
    } catch (exception *e) {
        cerr << "Error:\n\t" << e->what() << endl;
        return -1;
    }

    arax_exit();
    return 0;
} // main
