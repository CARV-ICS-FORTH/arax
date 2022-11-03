#ifndef ARAXLIB_MGR_HEADER
#define ARAXLIB_MGR_HEADER
#include "accelThread.h"
#include "arax_pipe.h"
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <thread>

typedef map<string, void *> Str2VpMap;
typedef set<arax_proc *> ProcSet;

class AraxLibMgr {
public:
    class KernelMap {
        friend std::ostream &operator << (std::ostream &os, KernelMap km);

public:
        void addKernel(std::string arch, std::string krnl, bool state);

private:
        std::map<std::string, std::map<std::string, bool> > krnl_state;
        std::set<std::string> krnl_archs;
    };
    AraxLibMgr();
    bool loadFolder(string lib_path, KernelMap &km, bool silent = false);
    bool loadLibrary(string lib_file, KernelMap &km);
    void unloadLibraries(arax_pipe_s *pipe);
    void startRunTimeLoader();
    ~AraxLibMgr();

private:
    static void rtld_handler(AraxLibMgr *libmgr);
    std::set<std::string> lib_paths;
    std::map<int, std::string> lib_fd_paths;
    Str2VpMap libs;
    ProcSet procs;
    std::thread rtld; // Run time loader
    int inotify_fd;
    bool run;
};

struct AraxProcedureDefinition
{
    const char *      name;
    arax_accel_type_e type;
    arax_accel_type_e max_type; /** Always set to ARAX_ACCEL_TYPES */
    AraxFunctor *     functor;
    size_t            arg_size;
};

#define ARAX_PROC_LIST_START()                                                 \
    extern "C" {                                                                 \
        struct AraxProcedureDefinition arax_proc_defs[] = {
#define ARAX_PROCEDURE(NAME, TYPE, FUNCTOR, ARG_SIZE)                          \
    { NAME, TYPE, ARAX_ACCEL_TYPES, (AraxFunctor *) FUNCTOR, ARG_SIZE },
#define ARAX_PROC_LIST_END()                                                   \
    { 0 }                                                                        \
    }                                                                            \
    ;                                                                            \
    }
#endif // ifndef ARAXLIB_MGR_HEADER
