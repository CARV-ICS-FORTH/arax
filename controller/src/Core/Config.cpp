#include "Config.h"
#include "../include/Formater.h"
#include "Scheduler.h"
#include "Utilities.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include "utils/system.h"

string trim(string &str)
{
    size_t start = str.find_first_not_of("\t ");
    size_t end   = str.find_last_not_of("\t ");

    if (start == string::npos)
        start = 0;
    if (end != string::npos)
        end += 1;
    return str.substr(start, end);
}

template <typename M, typename V>

void MapToVec(const M &m, V &v)
{
    for (typename M::const_iterator it = m.begin(); it != m.end(); ++it) {
        v.push_back(it->second);
    }
}

int GroupConfig::groupCount = 0;

GroupConfig::GroupConfig(std::string name, picojson::object conf) : name(name)
{
    try {
        picojson::object sched_obj;
        jsonGetSafeOptional(conf, "sched", sched_obj, " scheduler object ",
          sched_obj);

        std::string sched_class = "RoundRobin";
        jsonGetSafeOptional(sched_obj, "class", sched_class,
          " scheduler class string ", sched_class);

        scheduler = schedulerFactory.constructType(sched_class, sched_obj);

        if (!scheduler) {
            throw std::runtime_error(
                      std::string("Could not create scheduler of type '" + sched_class
                      + "' can be:[")
                      + join(schedulerFactory.getTypes()) + "]");
        }

        scheduler->setGroup(this);

        picojson::array accel_array;
        jsonGetSafe(conf, "accels", accel_array, " an array ");

        if (!accel_array.size())
            throw runtime_error("Groups accels array should not be empty");

        int accel_nr = 0;
        for (auto accel : accel_array) {
            picojson::object accel_obj;
            try {
                jsonCast(accel, accel_obj);

                new AccelConfig(accel_obj, this);
                accel_nr++;
            } catch (std::runtime_error &err) {
                throw std::runtime_error(std::string("While adding accelerator #")
                        + std::to_string(accel_nr) + " -> "
                        + err.what());
            }
        }
    } catch (std::runtime_error &err) {
        throw std::runtime_error(std::string("While parsing group '") + name
                + "' -> " + err.what());
    }

    groupId = groupCount;
    groupCount++;
}

int GroupConfig ::getCount(){ return groupCount; }

std::string GroupConfig ::getName() const { return name; }

int GroupConfig ::getID() const { return groupId; }

void GroupConfig ::addAccelerator(AccelConfig *accel)
{
    accelerators[accel->name] = accel;
    accel->group = this;
}

const map<string, AccelConfig *> &GroupConfig ::getAccelerators() const
{
    return accelerators;
}

size_t GroupConfig ::countAccelerators(AccelConfig::JobPreference pref)
{
    size_t count = 0;

    for (auto a : accelerators)
        count += a.second->job_preference == pref;
    return count;
}

size_t GroupConfig ::countAccelerators(){ return accelerators.size(); }

Scheduler * GroupConfig ::getScheduler(){ return scheduler; }

AccelConfig ::AccelConfig(picojson::object conf, GroupConfig *group)
    : conf(conf), type(ARAX_ACCEL_TYPES), arax_accel(0), accelthread(0),
    group(group)
{
    jsonGetSafe(conf, "arch", type_str, " accelerator arch string");
    type = arax_accel_type_from_str(type_str.c_str());

    std::string default_name = group->getName() + ":" + type_str + ":"
      + std::to_string(group->countAccelerators());

    jsonGetSafeOptional(conf, "name", name, "string", default_name);
    std::vector<std::string> a_types = threadFactory.getTypes();
    bool valid_accel_thread =
      std::count(a_types.begin(), a_types.end(), type_str + "accelThread");

    if (type == ARAX_ACCEL_TYPES || !valid_accel_thread) {
        for (auto &astr : a_types)
            astr = astr.substr(0, astr.size() - 11);
        throw runtime_error("Unknown accelerator type \'" + type_str
                + "\' at accelerator '" + name + "' should be:["
                + join(a_types) + "]");
    }

    std::string affinity;

    jsonGetSafeOptional(conf, "cpu_core", affinity, " thread core placement",
      std::string("-1"));
    std::istringstream iss(affinity);

    iss >> this->affinity;

    std::string job_pref;

    jsonGetSafeOptional(conf, "job_pref", job_pref,
      " accelerator job preference string",
      std::string("AnyJob"));

    group->addAccelerator(this);

    initial_preference = job_preference = fromString(job_pref, name);
}

CpuSet ::CpuSet(){ CPU_ZERO(&cpu_set); }

void CpuSet ::setAll()
{
    for (unsigned int core = 0; core < std::thread::hardware_concurrency();
      core++)
    {
        CPU_SET(core, &cpu_set);
    }
}

void CpuSet ::clearAll(){ CPU_ZERO(&cpu_set); }

void CpuSet ::set(int core)
{
    if (core >= 0) {
        if (core >= (int) std::thread::hardware_concurrency())
            throw runtime_error("Setting non existing core " + std::to_string(core));
        CPU_SET(core, &cpu_set);
    } else {
        setAll();
    }
}

void CpuSet ::clear(int core)
{
    if (core >= 0) {
        if (core >= (int) std::thread::hardware_concurrency())
            throw runtime_error("Clearing non existing core " + std::to_string(core));
        CPU_CLR(core, &cpu_set);
    } else {
        clearAll();
    }
}

cpu_set_t * CpuSet ::getSet(){ return &cpu_set; }

AccelConfig::JobPreference fromString(std::string pref, std::string loc)
{
    std::map<std::string, AccelConfig::JobPreference> pref_map;

    std::transform(pref.begin(), pref.end(), pref.begin(), ::tolower);

    pref_map["nojob"]    = AccelConfig::NoJob;
    pref_map["userjob"]  = AccelConfig::UserJob;
    pref_map["batchjob"] = AccelConfig::BatchJob;
    pref_map["anyjob"]   = AccelConfig::AnyJob;

    if (pref_map.count(pref) == 0) {
        throw runtime_error(
                  "Parse error at " + loc
                  + " expected a JobPreference(NoJob,UserJob,BatchJob,AnyJob)");
    }

    return pref_map.at(pref);
}

Config ::Config(string config_file)
{
    ifstream ifs(config_file);
    string line;
    string token;

    if (!ifs) {
        throw runtime_error("Config file \"" + config_file
                + "\" could not be read!");
    }

    picojson::value v;
    std::string err;

    const char *env_conf = system_env_var("ARAXCNTRL_CONF");

    if (env_conf)
        err = picojson::parse(v, env_conf, env_conf + strlen(env_conf));
    else
        err = picojson::parse(v, ifs);

    if (!err.empty()) {
        throw runtime_error("File \"" + config_file + "\" parse error: " + err);
    }

    picojson::object conf;

    jsonCast(v, conf);

    jsonGetSafeOptional(conf, "clean_shm", clean_shm, "bool", true);

    picojson::array jpaths;

    jsonGetSafeOptional(conf, "paths", jpaths, "array", jpaths);

    try {
        for (picojson::value path : jpaths) {
            std::string spath;

            jsonCast(path, spath);

            if (spath[0] == '/') {
                throw runtime_error("Only relative paths (not: '" + path.to_str()
                        + "')");
            }

            paths.push_back(spath);
        }
    } catch (std::runtime_error &err) {
        throw std::runtime_error(std::string("While parsing paths array->\n")
                + err.what());
    }

    try {
        picojson::object jgroups;

        jsonCast(conf["groups"], jgroups);

        if (jgroups.size() == 0) {
            throw runtime_error("File \"" + config_file
                    + "\" groups object should not be empty");
        }

        for (auto group : jgroups) {
            if (groups.count(group.first)) {
                throw runtime_error("File \"" + config_file + "\" group '"
                        + group.first + "' redeclared");
            }

            picojson::object grp;
            jsonGetSafe(jgroups, group.first, grp, "group object ({})");

            GroupConfig *ngc = new GroupConfig(group.first, grp);
            ngc->getScheduler()->setConfig(this);
            schedulers[ngc->getName()] = ngc->getScheduler();

            groups[group.first] = ngc;

            for (auto accel : ngc->getAccelerators())
                accelerators.emplace(accel);
        }
    } catch (std::runtime_error &err) {
        throw std::runtime_error(std::string("While parsing groups -> ")
                + err.what());
    }
}

const vector<string> Config ::getPaths() const { return paths; }

const AccelConfig * Config ::getAccelerator(std::string accel) const
{
    return accelerators.at(accel);
}

AccelConfig *&Config ::getAccelerator(std::string accel)
{
    return accelerators[accel];
}

const vector<AccelConfig *> Config ::getAccelerators() const
{
    vector<AccelConfig *> ret;

    ret.reserve(accelerators.size());
    for (auto accel : accelerators) {
        ret.push_back(accel.second);
    }
    if (ret.size() == 0) {
        std::cerr << __func__ << ": "
                  << " accelerator not found. Check your json file." << std::endl;
    }

    return ret;
}

#define MATCH                                                                  \
    [regex](const AccelConfig * p){ return std::regex_match(p->name, regex); }

const vector<AccelConfig *> Config ::getAccelerators(std::string name) const
{
    std::vector<AccelConfig *> vec;

    MapToVec(accelerators, vec);
    std::vector<AccelConfig *> ret;

    std::regex regex(name);

    ret.resize(std::count_if(vec.begin(), vec.end(), MATCH));
    std::copy_if(vec.begin(), vec.end(), ret.begin(), MATCH);
    return ret;
}

vector<AccelConfig *> Config ::getAccelerators(arax_accel_type_e type) const
{
    vector<AccelConfig *> ret;

    ret.reserve(accelerators.size());
    if (type != ANY) {
        for (auto accel : accelerators) {
            if (accel.second->type == type)
                ret.push_back(accel.second);
        }
    } else {
        for (auto accel : accelerators) {
            ret.push_back(accel.second);
        }
    }
    if (ret.size() == 0) {
        std::cerr << __func__ << ": " << arax_accel_type_to_str(type)
                  << " accelerator not found. Check your json file.\n";
    }
    return ret;
}

const vector<Scheduler *> Config ::getSchedulers() const
{
    vector<Scheduler *> vec;

    MapToVec(schedulers, vec);
    return vec;
}

const vector<GroupConfig *> Config ::getGroups() const
{
    vector<GroupConfig *> vec;

    MapToVec(groups, vec);
    return vec;
}

bool Config ::shouldCleanShm(){ return clean_shm; }

Config ::~Config()
{
    /*Free all maps*/
    for (map<string, AccelConfig *>::iterator it = accelerators.begin();
      it != accelerators.end(); ++it)
    {
        delete it->second;
    }

    for (map<string, Scheduler *>::iterator it = schedulers.begin();
      it != schedulers.end(); ++it)
    {
        delete it->second;
    }

    for (map<string, GroupConfig *>::iterator it = groups.begin();
      it != groups.end(); ++it)
    {
        delete it->second;
    }
}

istream &operator >> (istream &is, CpuSet &cpu_set)
{
    std::string fmt;
    int core;

    is >> fmt;
    std::replace(fmt.begin(), fmt.end(), ',', ' ');

    istringstream iss(fmt);

    do {
        iss >> core;
        if (iss)
            cpu_set.set(core);
    } while (iss);

    return is;
}

ostream &operator << (ostream &os, CpuSet &cpu_set)
{
    std::vector<int> cores;

    for (unsigned int core = 0; core < std::thread::hardware_concurrency();
      core++)
    {
        if (CPU_ISSET(core, cpu_set.getSet()))
            cores.push_back(core);
    }

    os << Formater<int>("", ",", "", cores);
    return os;
}

ostream &operator << (ostream &os, const AccelConfig *conf)
{
    CpuSet temp = conf->affinity;

    if (conf) {
        os << arax_accel_type_to_str(conf->type) << ", " << conf->name << ", "
           << temp << ", \""
           << "\"";
    } else {
        os << "Missing AccelConfig specification!" << endl;
    }
    return os;
}

ostream &operator << (ostream &os, const GroupConfig *conf)
{
    std::string sep = "";

    os << conf->getName() << "{";
    for (auto accel : conf->getAccelerators()) {
        os << sep << accel.first;
        sep = ",";
    }
    os << "}";
    return os;
}

ostream &operator << (ostream &os, const Config &conf)
{
    os << "Repository Paths:" << endl;
    os << Formater<string>("\t", "\n\t", "\n", conf.getPaths()) << endl;
    os << "Accelerators:" << endl;
    os << Formater<AccelConfig *>("\t", "\n\t", "\n", conf.getAccelerators());
    os << "Groups:" << endl;
    os << Formater<GroupConfig *>("\t", "\n\t", "\n", conf.getGroups());
    return os;
}
