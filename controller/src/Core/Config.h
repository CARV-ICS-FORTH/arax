#ifndef CONFIG_FILE_HEADER
#define CONFIG_FILE_HEADER
using namespace ::std;

class accelThread;
class GroupConfig;
class Config;

#include "Utilities.h"
#include <arax_pipe.h>
#include <core/arax_accel_types.h>
#include <map>
#include <string>
#include <vector>

/*
 * CpuSet class
 */
class CpuSet {
public:
    CpuSet();
    void setAll();
    void clearAll();
    void set(int core);
    void clear(int core);
    cpu_set_t* getSet();

private:
    cpu_set_t cpu_set;
};

/*
 * Struct that describes the accelerators of one node
 *  the accelerators are described in a configuration file
 */
struct AccelConfig
{
    typedef enum { NoJob, UserJob, BatchJob, AnyJob } JobPreference;

    AccelConfig(picojson::object conf, GroupConfig *group);
    string            name;           // Accelerator Name
    picojson::object  conf;           // For Gpus PCI id
    string            type_str;       // type string will fix this.
    arax_accel_type_e type;           // Accelerator type GPU, CPU, FPGA
    CpuSet            affinity;       // Thread affinity
    arax_accel_s *    arax_accel;     // Arax accelerator object
    accelThread *     accelthread;    // Accelerator Thread
    GroupConfig *     group;          // Group where this accelerator belongs
    Config *          config;         // System configuration
    JobPreference     job_preference; // What jobs this accelerator accepts
    JobPreference
                      initial_preference; // initial job preference that accelerator accepts
};

#include <Core/Scheduler.h>

AccelConfig::JobPreference fromString(std::string, std::string);

/*
 * Struct that describes a group of accelerators a group contains accelerators
 * with similar specs that are handled from the same Scheduler
 */
class GroupConfig {
public:
    GroupConfig(std::string name, picojson::object conf);
    static int getCount();
    std::string getName() const;
    int getID() const;
    void addAccelerator(AccelConfig *accel);
    const map<string, AccelConfig *> &getAccelerators() const;
    size_t countAccelerators(AccelConfig::JobPreference pref);
    size_t countAccelerators();
    Scheduler* getScheduler();

private:
    std::string name;
    Scheduler *scheduler;                    // Scheduler handling this group
    map<string, AccelConfig *> accelerators; // Accelerators in this group
    /*Id per group*/
    int groupId;
    static int groupCount;
};

/*
 * Classs that describes the configuration file with
 *  1. The paths of .so files
 *  2. The accelerators
 *  3. The schedulers
 *  4. The groups of accels
 */
class Config {
public:
    Config(string config_file);
    string getRepository();
    const vector<string> getPaths() const;
    const AccelConfig* getAccelerator(std::string accel) const;
    AccelConfig *&getAccelerator(std::string accel);
    const vector<AccelConfig *> getAccelerators() const;
    const vector<AccelConfig *> getAccelerators(std::string regex) const;
    vector<AccelConfig *> getAccelerators(arax_accel_type_e type) const;
    const vector<Scheduler *> getSchedulers() const;
    const vector<GroupConfig *> getGroups() const;
    bool shouldCleanShm();
    ~Config();

private:
    bool clean_shm;
    vector<string> paths;                    // Paths to load libraries from
    map<string, AccelConfig *> accelerators; // Accelerators defined in config
    map<string, Scheduler *> schedulers;     // Schedulers defined in config
    map<string, GroupConfig *> groups;       // Groups defined in config
};

istream &operator >> (istream &is, CpuSet &cpu_set);
ostream &operator << (ostream &os, CpuSet &cpu_set);
ostream &operator << (ostream &os, const Config &conf);
ostream &operator << (ostream &os, const AccelConfig *conf);
ostream &operator << (ostream &os, const GroupConfig *conf);
#endif // ifndef CONFIG_FILE_HEADER
