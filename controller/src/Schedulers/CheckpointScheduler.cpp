#include "CheckpointScheduler.h"
#include <iostream>
// #include "core/arax_data.h"
using namespace ::std;

CheckpointScheduler::CheckpointScheduler(string args){ }

CheckpointScheduler::~CheckpointScheduler(){ }

extern arax_pipe_s *vpipe_s;
void CheckpointScheduler::checkpointAllActiveTasks(accelThread *th)
{
    cerr << "----- Start with Checkpoint!! -----" << endl;

    // Get all the data allocated in ALL accelerators
    utils_list_s *araxDataList =
      arax_object_list_lock(&(vpipe_s->objs), ARAXTYPE_DATA);
    utils_list_node_s *vacNode;
    arax_data_s *vdata;
    // Get the pysicall accelerator of the current thread
    arax_accel_s *threadPhysAccel = th->getAccelConfig().arax_accel;

    // cerr<<"Create the vector of vdata"<<endl;

    utils_list_for_each(*araxDataList, vacNode){
        // get a arax_data of that node
        vdata = (arax_data_s *) vacNode->owner;

        // get the pysical accel of arax_data
        arax_accel *dataPhysAccel =
          (arax_accel *) (((arax_vaccel_s *) (vdata->accel))->phys);
        bool remote_vData = arax_data_has_remote(vdata);

        // if data belongs to this accelThread
        if ((dataPhysAccel == threadPhysAccel) && (remote_vData)) {
            // Add arax data to vector
            araxDataCH.push_back(vdata);
            // cerr<<"AraxData: "<<vdata<<endl;
        }
    }
    // cerr<<"Done with the vector of vdata"<<endl;
    arax_object_list_unlock(&(vpipe_s->objs), ARAXTYPE_DATA);

    // Perform Checkpoint for all arax data
    cerr << "Start transfering data to shm: " << araxDataCH.size() << endl;
    for (auto iter = araxDataCH.begin(); iter != araxDataCH.end(); ++iter) {
        // cerr<<"(From Vector) AraxData: "<< *iter<<endl;
        arax_vaccel_s *dataVirtAccel = ((arax_vaccel_s *) (**iter).accel);
        arax_data_shm_sync(dataVirtAccel, "syncFrom", *iter, 0);
    }
    araxDataCH.clear();
    araxDataCH.shrink_to_fit();
    // XXX TODO XXX freeRemote

    cerr << "----- Done with Checkpoint!! -----" << endl;
} // CheckpointScheduler::checkpointAllActiveTasks

/* Determines the frequency of the checkpoint*/
void CheckpointScheduler::checkpointFrequency(accelThread *th){ }

/*Takes a snapshot for one task*/
void CheckpointScheduler::checkpoint1Task(arax_task_msg_s *arax_task,
  accelThread *                                            th)
{
    /*Use arax_object_list_lock from core/arax_object.h*/

    /*
     * arax_data_s *data;
     * cerr<<"START: Checkpoint task ( "<<arax_task<<" )"<<endl;
     * for (int arg = arax_task->in_count; arg <
     * arax_task->in_count+arax_task->out_count; arg++)
     * {
     *  data = (arax_data_s *)arax_task->io[arg];
     *  size_t sz = arax_data_size(data);
     *  //cerr<<"Size: "<<sz<<endl;
     *  if(data->remote)
     *  {
     *      //  cerr<<"data to remote!!"<<endl;
     *      cudaError_t err = cudaMemcpy(arax_data_deref(data),
     *              data->remote, sz, cudaMemcpyDeviceToHost);
     *
     *      if( err != cudaSuccess ){
     *          cerr << " CUDA error : " << cudaGetErrorString(err) <<"
     * "<<__FILE__<< endl; return false;
     *
     *      }
     *
     *  }
     * }
     */
    cerr << "checkpoint1Task is not implemented!! " << endl;
}

void CheckpointScheduler::setGroup(GroupConfig *group){ this->group = group; }

void CheckpointScheduler::setCheckFreq(arax_task_msg_s *task){ }

void CheckpointScheduler::resetCheckFreq(){ }

Factory<CheckpointScheduler, std::string> checkpointSchedulerFactory;
