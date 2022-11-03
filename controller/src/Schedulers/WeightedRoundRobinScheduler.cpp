#include "definesEnable.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>
using namespace ::std;
#include "WeightedRoundRobinScheduler.h"
/*ClientState Constructor*/
WeightedRoundRobinScheduler::ClientState::ClientState(vector<int> weight)
    : vectorCredit(weight), vectorWeight(weight) {}
WeightedRoundRobinScheduler::ClientState::~ClientState() {}
/*State Constructor*/
WeightedRoundRobinScheduler::State::State() {
  utils_queue_init(&low);
  utils_queue_init(&high);
}
/*State Constructor*/
WeightedRoundRobinScheduler::State::~State() {}
/*JobDescription Constructor*/
WeightedRoundRobinScheduler::JobDescription::JobDescription() {}
/*JobDescription Constructor*/
WeightedRoundRobinScheduler::VAQInfo::VAQInfo(int numOfGroups) : inGroup(0) {}
/*WeightedRoundRobinScheduler Constructor*/
WeightedRoundRobinScheduler::WeightedRoundRobinScheduler(
    picojson::object args, WeightedRoundRobinScheduler::State *state)
    : Scheduler(args), state(state) {
  loadConfigurationFile();
#ifdef REDUCTION_UNITS_FROM_EXEC_TIME
  loadreductionUnitFile();
#endif
}
/*WeightedRoundRobinScheduler Destructor*/
WeightedRoundRobinScheduler::~WeightedRoundRobinScheduler() {}
/*Select a Virtual Accelerator Queue from all the VAQs that exists in the
 * system*/
utils_queue_s *
WeightedRoundRobinScheduler::selectVirtualAcceleratorQueue(accelThread *th) {
  cout << "Not Used" << endl;
  return 0;
}
/*
 * Select a VAQ to put it in LOW/HIGH queue
 * Inputs : An array with all VAQs of the system,
 * 	    The size of the array with all VAQs,
 * 	    The type of the physical accelerator
 * Outputs: Returns a LOW or HIGH priority control queue (arax_vaccel).
 *
 */
arax_vaccel_s *
WeightedRoundRobinScheduler::selectVirtualAcceleratorQueue(accelThread *th,
                                                           size_t id) {
  /*init variables*/
  arax_accel **arrayAllVAQs = th->getAllVirtualAccels();
  int numOfVAQs = th->getNumberOfVirtualAccels();
  AccelConfig &accelConf = th->getAccelConfig();
  /*Struct with all the information about a client - VAQ pair*/
  VAQInfo *vaqInfo = 0;
  /*the returned VAQ*/
  arax_vaccel_s *araxVAccelSelected = 0;
  /*Defines if something is inserted to a LOW or HIGH Cntrl queue*/
  bool insertedSomething = false;
  /***Step 1 : Scan all VAQs (till numOfVAQs), and if needed insert them into
   * HIGH/LOW control queues***/
  for (int q = 0; q < numOfVAQs; q++) {
    vaqInfo = (WeightedRoundRobinScheduler::VAQInfo *)arax_vaccel_get_meta(
        (arax_vaccel_s *)(arrayAllVAQs[q]));
    if (!vaqInfo || (vaqInfo->getInGroup(accelConf.group->getID()) == false)) {
      /*retrieve customers Id*/
      id = getId((arax_vaccel_s *)(arrayAllVAQs[q]));
      /*check the job id*/
      checkJobIdRange(id);
      /*Get client weight*/
      int customerWeight = getClientWeight(id, accelConf.group->getID());
      if (customerWeight == 0) {
        // cout<<accelConf.group->getID()<<" Weight is 0!!!!!!!!!"<<endl;
        continue;
      }
      /*Create a vaqInfo Obj if the VAQ is newly created*/
      if (!vaqInfo) {
        vaqInfo = new VAQInfo(GroupConfig::getCount());
      }
      /*VAQ already exists and has VAQ info*/
      vaqInfo->setInGroup(true, accelConf.group->getID());
      /*Customer ID based on the VAQ number (passed to the VAQ information).*/
      vaqInfo->clientId = id;
      /*Get the selected utils_queue_s of the selected VAQ */
      vaqInfo->utilsQueue =
          arax_vaccel_queue((arax_vaccel_s *)(arrayAllVAQs[q]));
      /*if the selected utils_queue contains tasks*/
      if (utils_queue_used_slots(vaqInfo->utilsQueue)) {
        /*Set VAQ info*/
        arax_vaccel_set_meta((arax_vaccel_s *)(arrayAllVAQs[q]), vaqInfo);
      }
      /*The selected utils_queue is empty*/
      else {
        vaqInfo->setInGroup(false, accelConf.group->getID());
        if (!vaqInfo->getInAnyGroup()) {
          arax_vaccel_set_meta((arax_vaccel_s *)(arrayAllVAQs[q]), 0);
          if (vaqInfo)
            delete vaqInfo;
        }
        continue;
      }
      /*VAQ has positive number of credits*/
      if (getClientCredits(vaqInfo->clientId, accelConf.group->getID()) > 0) {
        /*Add the selected VAQ to HIGH control queue*/
        utils_queue_push(&(state->high), (arax_vaccel_s *)(arrayAllVAQs[q]));
        // cout<<__LINE__<<":  Push In HIGH VAQ->"<<(void
        // *)(arrayAllVAQs[q])<<endl;
      } else {
        /*Add the selected VAQ to LOW control queue*/
        utils_queue_push(&(state->low), (arax_vaccel_s *)(arrayAllVAQs[q]));
      }
      insertedSomething = true;
    }
  }
  /******* End of Step 1 ********/
  /*
   * If nothing is inserted to LOW/HIGH priority queues and there are VAQs
   * this means that no Task has been executed.
   */
  if (!insertedSomething && numOfVAQs) {
    /*Increase the semaphore that has been reduced due to wait_for_task*/
    arax_pipe_add_task(th->getPipe(), accelConf.type, th);
  }
  /****Step 2 : Select a VAQ from HIGH/LOW control queues*****/
  {
    /*Specifies from which queue (LOW/HIGH) we pop*/
    // int queue = 0 ;
    /*Weight of the a specific Job/client*/
    int customerWeight;
    /*If HIGH contrl queue  has VAQs, choose HIGH as a controll queue*/
    if (utils_queue_used_slots(&(state->high))) {
      /*Specifies from which I am going to pop*/
      // queue = 60;
      /*
       * If High queue is non empty and customer is able to be executed in a
       * Group (its weight is !=0) keep pop VAQs from HIGH queue.
       */
      do {
        /*Pop from High*/
        araxVAccelSelected = (arax_vaccel_s *)utils_queue_pop(&(state->high));
        /*Get the meta from that VAQ*/
        int customerid = getId(araxVAccelSelected);
        /*check the job id*/
        checkJobIdRange(customerid);
        /*get the job/client weight*/
        customerWeight = getClientWeight(customerid, accelConf.group->getID());
      } while (customerWeight == 0 && utils_queue_used_slots(&(state->high)));
      /*High queue is empty*/
      if (!araxVAccelSelected)
        return 0;
      /*Get meta from the selected VAQ*/
      vaqInfo = (WeightedRoundRobinScheduler::VAQInfo *)(arax_vaccel_get_meta(
          araxVAccelSelected));
      /*The selected VAQ is empty*/
      if (utils_queue_used_slots(vaqInfo->utilsQueue) == 0) {
        /*Increase the semaphore since no task has been executed*/
        arax_pipe_add_task(th->getPipe(), accelConf.type, th);
        vaqInfo->setInGroup(false, accelConf.group->getID());
        myMutex.unlock();
        return 0;
      }
#ifndef REDUCTION_UNITS_FROM_EXEC_TIME
      /*Decrease the credits of a particular Job/Customer*/
      setClientCredits(vaqInfo->clientId, accelConf.group->getID(), -1);
#else
      /*Peek the arax_task, not reduce the semaphore*/
      arax_task_msg_s *araxTaskTmp = (arax_task_msg_s *)utils_queue_peek(
          arax_vaccel_queue(araxVAccelSelected));
      /*Get the relevant reduction unit for that task type*/
      int reduction_unit = 0;
      reduction_unit =
          reductionUnitsForAllAraxProcs[((arax_object_s *)araxTaskTmp->proc)
                                            ->name]
              .RUnitsVector[accelConf.group->getID()];
      /*Decrease the credits of a particular Job/Customer*/
      setClientCredits(vaqInfo->clientId, accelConf.group->getID(),
                       -reduction_unit);
#endif
    }
    /*If HIGH is empty go to LOW*/
    else {
      // cout<<__LINE__<<"HIGH is empty!! GO to LOW"<<endl;
      /*Pop from Low queue*/
      // queue = 30 ;
      /*If LOW is also empty, there is no VAQ to serve*/
      if (utils_queue_used_slots(&(state->low)) == 0) {
        myMutex.unlock();
        return 0;
      }
      /*
       * If Low queue is non empty and customer is able to be executed in a
       * Group (its weight is !=0) keep pop VAQs from LOW queue.
       */
      do {
        /*pop task from LOW queue*/
        araxVAccelSelected = (arax_vaccel_s *)utils_queue_pop(&(state->low));
        /*Get the meta of the selected VAQ*/
        vaqInfo = (WeightedRoundRobinScheduler::VAQInfo *)(arax_vaccel_get_meta(
            araxVAccelSelected));
        /*Retrieve job/client id from the selected VAQ*/
        int customerid = getId(araxVAccelSelected);
        /*check the job id*/
        checkJobIdRange(customerid);
        /*Get Job/client weight*/
        customerWeight = getClientWeight(customerid, accelConf.group->getID());
      } while (customerWeight == 0 && utils_queue_used_slots(&(state->low)));
      /*if there is no VAQ in LOW return 0*/
      if (!araxVAccelSelected) {
        myMutex.unlock();
        return 0;
      }
      /*Get the meta of the selected VAQ*/
      vaqInfo = (WeightedRoundRobinScheduler::VAQInfo *)(arax_vaccel_get_meta(
          araxVAccelSelected));
      /*The selected VAQ is empty*/
      if (utils_queue_used_slots(vaqInfo->utilsQueue) == 0) {
        /*Increase the semaphore since no task has been executed*/
        arax_pipe_add_task(th->getPipe(), accelConf.type, th);
        vaqInfo->setInGroup(false, accelConf.group->getID());
        myMutex.unlock();
        return 0;
      }
#ifndef REDUCTION_UNITS_FROM_EXEC_TIME
      /*Decrease the credits of that client by a reduction unit*/
      vectorClientState[vaqInfo->clientId]
          .vectorCredit[accelConf.group->getID()] -= 1;
#else
      /*Peek a arax_task without execute it (i.e. not decrease the semaphore)*/
      arax_task_msg_s *araxTaskTmp = (arax_task_msg_s *)utils_queue_peek(
          arax_vaccel_queue(araxVAccelSelected));
      int reduction_unit = 0;
      reduction_unit =
          reductionUnitsForAllAraxProcs[((arax_object_s *)araxTaskTmp->proc)
                                            ->name]
              .RUnitsVector[accelConf.group->getID()];
      /*decrease credits*/
      setClientCredits(vaqInfo->clientId, accelConf.group->getID(),
                       -reduction_unit);
#endif
      /*increase VAQ's for this scheduler, credit by the client's weight*/
      setClientCredits(
          vaqInfo->clientId, accelConf.group->getID(),
          getClientWeight(vaqInfo->clientId, accelConf.group->getID()));
    }
    //	time_t rawtime;
    //  	struct tm * timeinfo;
    //  	char buffer [256];
    //	time (&rawtime);
    //	timeinfo = localtime (&rawtime);
    if (vectorClientState.size() == 4) {
      //		strftime(buffer,sizeof(buffer),"%Y/%m/%d
      //%H:%M:%S",timeinfo); 		string str(buffer) ;
      /*cout <<" AccelGroup: "<<accelConf.group->getID()
        <<" -Credit_CL_0: "<<
      vectorClientState[0].vectorCredit[accelConf.group->getID()]
        <<" -Credit_CL_1: "<<
      vectorClientState[1].vectorCredit[accelConf.group->getID()]
        <<" -Credit_CL_2: "<<
      vectorClientState[2].vectorCredit[accelConf.group->getID()]
        <<" -Credit_CL_3: "<<
      vectorClientState[3].vectorCredit[accelConf.group->getID()]
        <<" -Pop from: "   << queue
      //<<" -Time: " << str
      <<endl;
      */
    } else {
      cout << "Please check customer Weights file. And add zeros for non "
              "existing clients!"
           << endl;
    }
  }
  myMutex.unlock();
  return araxVAccelSelected;
}
/**
 * Pop a task from the selected VAQ.
 * Inputs : Array with all the VAQs exist in the system
 * 	    The size of the array
 * 	    The physical accelerator
 * Outputs : A selected task
 *
 * */
arax_task_msg_s *WeightedRoundRobinScheduler::selectTask(accelThread *th) {
  /*The customerId of a VAQ*/
  size_t customer_id = 0;
  int id = 0;
  /*Task poped from that VAQ*/
  arax_task_msg_s *arax_task;
  /*VAQ that the task is going to be poped*/
  arax_vaccel_s *selectedVAQ = 0;
  myMutex.lock();
  /*Choose a VAQ from all that exist in the system*/
  selectedVAQ = selectVirtualAcceleratorQueue(th, id);
  /*If the selected VAQ = 0, then no VAQ eligible*/
  if (selectedVAQ == 0) {
    myMutex.unlock();
    return 0;
  }
  /*Get the customer ID of the selected arax_vaccel_s*/
  customer_id = arax_vaccel_get_cid(selectedVAQ);
  checkJobIdRange(customer_id);
  /*Take the task from the selected VAQ*/
  arax_task =
      (arax_task_msg_s *)utils_queue_pop(arax_vaccel_queue(selectedVAQ));
  // cout <<"QUEUE "<<selectedVAQ<<"
  // "<<utils_queue_used_slots(arax_vaccel_queue(selectedVAQ))<<endl;
  // cout<<__LINE__<<":  (selectTask)  Used slots : "<<
  // utils_queue_used_slots(arax_vaccel_queue(selectedVAQ))<<"   customer ID:
  // "<< customer_id<<endl;
  /*get meta for the selected VAQ*/
  VAQInfo *meta = (VAQInfo *)arax_vaccel_get_meta(selectedVAQ);
  /*Check if there are meta (change it with while)*/
  if (!meta) {
    cout << "No meta!" << endl;
    meta = (VAQInfo *)arax_vaccel_get_meta(selectedVAQ);
  }
  /*The selected VAQ is empty so remove it from Low of High Queue*/
  if (utils_queue_used_slots(arax_vaccel_queue(selectedVAQ)) == 0) {
    // VAQInfo *meta = (VAQInfo *) arax_vaccel_get_meta(selectedVAQ);
    /*The selected VAQ is empty so it should not be in any group*/
    meta->setInGroup(false, th->getAccelConfig().group->getID());
    if (!meta->getInAnyGroup()) {
      /*Set meta to 0*/
      arax_vaccel_set_meta(selectedVAQ, 0);
      if (meta)
        delete meta;
    }
  }
  /*There are more tasks in the selected VAQ */
  else {
    /*
     * If client's weight has become different than 0 in a group
     * this group is enabled to execute
     */
    if (getClientWeight(meta->clientId, th->getAccelConfig().group->getID()) >
        0) {
      cout << __LINE__ << " GROUP id: " << th->getAccelConfig().group->getID()
           << " Weight: "
           << getClientWeight(meta->clientId,
                              th->getAccelConfig().group->getID())
           << endl;
      /*Credits are positive so add VAQ to HIGH priority Queue*/
      if (getClientCredits(customer_id, th->getAccelConfig().group->getID()) >
          0) {
        /*PUSH the selected VAQ to the HIGH queue*/
        utils_queue_push(&(state->high), (selectedVAQ));
      }
      /*Credits are negative so add VAQ to LOW priority Queue*/
      else {
        /*PUSH the selected VAQ to the LOW queue*/
        utils_queue_push(&(state->low), (selectedVAQ));
      }
    }
    /*client's weight has become 0 in a group (group is disabled)*/
    else {
      cerr << __LINE__ << " Client Weight <0 !!!!!!!!!!!!!!!!!!" << endl;
      /*remove this VAQ from Low High Queue*/
      meta->setInGroup(false, th->getAccelConfig().group->getID());
      if (!meta->getInAnyGroup()) {
        /*Set meta to 0*/
        arax_vaccel_set_meta(selectedVAQ, 0);
        if (meta)
          delete meta;
      }
    }
  }
  arax_task->accel = selectedVAQ;
  myMutex.unlock();
  return arax_task;
}
/*Load the configuration file with the weights of the customers*/
void WeightedRoundRobinScheduler::loadConfigurationFile() {
  ifstream fin("customerWeights");
  if (!fin) {
    cerr << "customerWeights does NOT exists !" << endl;
    cerr << "Please create it in the arax_controller directory." << endl;
    exit(-1);
  }
  string line;
  while (getline(fin, line)) {
    if (line[0] == '#')
      continue;
    stringstream ss(line);
    vector<int> weightsPerClient;
    int clientWeight;
    do {
      ss >> clientWeight;
      if (ss)
        weightsPerClient.push_back(clientWeight);
    } while (ss);
    ClientState stateTmp(weightsPerClient);
    vectorClientState.push_back(stateTmp);
  }
  /*print .customerWeights file*/
  /*
     vector<ClientState>::iterator it_c;
     vector<int>::iterator it_w;
     int i = 0 ;
     for (it_c = vectorClientState.begin(); it_c != vectorClientState.end();
     ++it_c, ++i)
     {
     for (it_w = it_c->vectorWeight.begin(); it_w != it_c->vectorWeight.end();
     ++it_w)
     {
     cout <<"Client: " << i << " with weight " << *it_w << endl;
     }
     }
     */
  fin.close();
}
/* Load the credits for each job.
 *   This file is provided by Job Generator
 */
void WeightedRoundRobinScheduler::loadreductionUnitFile() {
  cout << "Load Reduction units file" << endl;
  ifstream fin("reductionUnit");
  if (!fin) {
    cerr << __LINE__ << ": reductionUnit file does NOT exists !" << endl;
    cerr << "Please create it in the arax_controller directory." << endl;
    exit(-1);
  }
  string line;
  /*Create a Job Description instance that will contain all info for a job*/
  while (getline(fin, line)) {
    if (line[0] == '#')
      continue;
    stringstream ss(line);
    /*Get the job name*/
    string jobName;
    JobDescription JobDesc;
    ss >> jobName;
    /*Get the Execution time for each Job from the file provided bu JG*/
    while (ss) {
      float tmp;
      /*Get the execution time for each Job*/
      ss >> tmp;
      if (ss) {
        JobDesc.RUnitsVector.push_back(round(tmp));
      }
    }
    /*Add the JobDescription object into the reductionUunit map (arax_proc*,
     * JobDescription)*/
    arax_proc *proc;
    /*Check the existance of the particular procedure*/
    if ((proc = arax_proc_get(ANY, jobName.c_str()))) {
      /*Add the execution time to a map*/
      reductionUnitsForAllAraxProcs[jobName.c_str()] = JobDesc;
    } else {
      cerr << __LINE__ << ": arax procedure does not exists: " << jobName
           << endl;
      exit(-1);
    }
  }
  fin.close();
  /*Calculate the reduction units according to the execution time*/
  time2ReductionUnits();
  cout << "*****************************************************" << endl;
  /*Print the reduction unit calculated per Job type*/
  map<string, JobDescription>::iterator it;
  for (it = reductionUnitsForAllAraxProcs.begin();
       it != reductionUnitsForAllAraxProcs.end(); ++it) {
    cout << " " << it->first << "  ";
    vector<int>::iterator it_v;
    for (it_v = it->second.RUnitsVector.begin();
         it_v != it->second.RUnitsVector.end(); ++it_v) {
      cout << "" << *it_v << "  ";
    }
    cout << endl;
  }
  cout << "*****************************************************" << endl;
}
void WeightedRoundRobinScheduler::time2ReductionUnits() {
  if (reductionUnitsForAllAraxProcs.size() == 0) {
    cerr << __LINE__ << ": size of reductionUnits map is Null" << endl;
    cerr << "The sysyem will exit ..." << endl;
    exit(-1);
  }
  /*Find the maximum per column*/
  map<int, int> maxs;
  map<string, JobDescription>::iterator it1;
  for (it1 = reductionUnitsForAllAraxProcs.begin();
       it1 != reductionUnitsForAllAraxProcs.end(); ++it1) {
    int i = 0;
    for (auto c : it1->second.RUnitsVector) {
      maxs[i] = max(maxs[i], c);
      i++;
    }
  }
  /*Calculate the reduction units,
   * the maximum value will get 100 as a reduction unit
   * all the other values that are lower will be normalized
   * according to the maximum
   * */
  map<string, JobDescription>::iterator it2;
  for (it2 = reductionUnitsForAllAraxProcs.begin();
       it2 != reductionUnitsForAllAraxProcs.end(); ++it2) {
    //	cout<<it2->first<<" ";
    for (size_t i = 0;
         i < reductionUnitsForAllAraxProcs.begin()->second.RUnitsVector.size();
         i++) {
      it2->second.RUnitsVector[i] *= 100;
      it2->second.RUnitsVector[i] /= maxs[i];
      if (it2->second.RUnitsVector[i] == 0)
        it2->second.RUnitsVector[i] = 1;
      //		cout<<"  "<< it2->second.RUnitsVector[i];
    }
  }
}
void WeightedRoundRobinScheduler::checkJobIdRange(size_t jobId) {
  if (jobId > vectorClientState.size()) {
    cerr << __LINE__ << ": Job id is out of range! " << endl;
    cerr << "Please check set_cid() in jobGen!" << endl;
    exit(-1);
  }
}
/*Get client/tenant Id*/
int WeightedRoundRobinScheduler::getId(arax_vaccel_s *vaq) {
  int id = arax_vaccel_get_cid(vaq);
  while (id == -1) {
    cout << "ID = -1 . Retry!" << endl;
    id = arax_vaccel_get_cid(vaq);
  }
  return id;
}
/*Set credits of a client*/
void WeightedRoundRobinScheduler::setClientCredits(int clientId, int groupId,
                                                   int value) {
  vectorClientState[clientId].vectorCredit[groupId] += value;
  // cout<<"SetClientCredits for client: "<< clientId<<"  of groupid
  // "<<groupId<< "  with value "<<value<<endl;
}
/*Set weight of a client*/
void WeightedRoundRobinScheduler::setClientWeight(int clientId, int groupId,
                                                  int value) {
  vectorClientState[clientId].vectorWeight[groupId] = value;
}
/*Get credits of a client*/
int &WeightedRoundRobinScheduler::getClientCredits(int clientId, int groupId) {
  return vectorClientState[clientId].vectorCredit[groupId];
}
/*Get weights of a client*/
int &WeightedRoundRobinScheduler::getClientWeight(int clientId, int groupId) {
  return vectorClientState[clientId].vectorWeight[groupId];
}
REGISTER_SCHEDULER(WeightedRoundRobinScheduler)
