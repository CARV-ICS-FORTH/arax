#include "Utilities.h"
#include <algorithm>
#include <pthread.h>
#include <sstream>

template <> std::string jsonTypeToString(std::string){ return "string"; }

template <> std::string jsonTypeToString(picojson::object)
{
    return "JSON object ({})";
}

template <> std::string jsonTypeToString(picojson::array)
{
    return "JSON array ([])";
}

template <> std::string jsonTypeToString(int64_t){ return "integer"; }

template <> std::string jsonTypeToString(double){ return "float"; }

template <> std::string jsonTypeToString(bool){ return "boolean"; }

std::map<std::string, std::string> decodeArgs(std::string args)
{
    std::string k, v;
    std::istringstream iss(args);
    std::map<std::string, std::string> kv;

    do {
        std::getline(iss, k, ':');
        std::getline(iss, v, ',');
        if (iss) {
            k = trim(k);
            std::transform(k.begin(), k.end(), k.begin(), ::tolower);
            v     = trim(v);
            kv[k] = v;
        }
    } while (iss);

    return kv;
}

std::string join(const std::vector<std::string> &strs)
{
    std::string sep = "";
    std::string ret;

    for (auto &str : strs) {
        ret += sep + str;
        sep  = ", ";
    }
    return ret;
}

std::string trim(const std::string &s)
{
    auto wsfront = std::find_if_not(s.begin(), s.end(),
        [](int c){
        return std::isspace(c);
    });
    auto wsback = std::find_if_not(s.rbegin(), s.rend(), [](int c){
        return std::isspace(c);
    }).base();

    return (wsback <= wsfront ? std::string() : std::string(wsfront, wsback));
}

void set_thread_name(std::string name)
{
    pthread_setname_np(pthread_self(), name.c_str());
}
