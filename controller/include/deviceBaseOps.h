#ifndef DEVICE_BASE_OPS_HEADER
#define DEVICE_BASE_OPS_HEADER
#include "AraxLibMgr.h"
#include "core/arax_data.h"
/**
 * Basic device operations interface
 */
struct deviceBaseOps {
  /* Function that initializes an accelerator */
  virtual bool acceleratorInit() = 0;
  /* Function that resets an accelerator */
  virtual void acceleratorRelease() = 0;
  /*Executes an operation (syncTo, SyncFrom, free, kernel)*/
  virtual void executeOperation(AraxFunctor *functor,
                                arax_task_msg_s *task) = 0;
  /* Performs a 'reset' */
  virtual void reset(accelThread *) = 0;

  virtual bool alloc_no_throttle(arax_data_s *data) = 0;
  virtual void alloc_remote(arax_data_s *vdata) = 0;
  virtual void sync_to_remote(arax_data_s *vdata) = 0;
  virtual void sync_from_remote(arax_data_s *vdata) = 0;
  virtual void free_remote(arax_data_s *vdata) = 0;
};

#define IMPLEMENTS_DEVICE_BASE_OPS()                                           \
  virtual bool acceleratorInit();                                              \
  virtual void acceleratorRelease();                                           \
  virtual void executeOperation(AraxFunctor *functor, arax_task_msg_s *task);  \
  virtual void reset(accelThread *);                                           \
  virtual bool alloc_no_throttle(arax_data_s *data);                           \
  virtual void alloc_remote(arax_data_s *vdata);                               \
  virtual void sync_to_remote(arax_data_s *vdata);                             \
  virtual void sync_from_remote(arax_data_s *vdata);                           \
  virtual void free_remote(arax_data_s *vdata);

#define USES_DEFAULT_EXECUTE_HOST_CODE(CLASS)                                  \
  void CLASS::executeOperation(AraxFunctor *functor,                           \
                               arax_task_msg_s *arax_task) {                   \
    try {                                                                      \
      functor(arax_task);                                                      \
    } catch (std::exception & e) {                                             \
      cout << "AraxFunctor !!! " << e.what() << endl;                          \
    }                                                                          \
  }

/**
 * Provides a No Op reset implementation.
 */
#define USES_NOOP_RESET(CLASS)                                                 \
  void CLASS::reset(accelThread *) {}

#endif
