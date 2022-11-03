#ifndef ARAXCONTROLLER_UTILITIES_HEADER_FILE
#define ARAXCONTROLLER_UTILITIES_HEADER_FILE
#include <map>
#include <string>
#define PICOJSON_USE_INT64
#include <picojson.h>

/**
 * Convert arguement list of the following format to a map:
 *
 * key1:val1,key2:val2
 *
 * Resulting map has keys and values in lowercase.
 * Keys and values are assumed to be words
 * (i.e a key/value can not contain whitespace)
 */
std::map<std::string, std::string> decodeArgs(std::string args);

template <class C> std::string jsonTypeToString(C)
{
    return "Unexpected JSON type!";
}

template <class C> void jsonCast(picojson::value &val, C &casted)
{
    if (!val.is<C>()) {
        throw std::runtime_error("Wanted '" + jsonTypeToString(casted)
                + "' instead got " + val.to_str());
    }
    casted = val.get<C>();
}

template <class C>
void jsonGetSafe(picojson::object &obj, std::string key, C &val,
  std::string type)
{
    try {
        jsonCast(obj[key], val);
    } catch (std::runtime_error &err) {
        throw std::runtime_error("While accessing key '" + key + "' -> "
                + err.what());
    }
}

template <class C>
void jsonGetSafeOptional(picojson::object &obj, std::string key, C &val,
  std::string type, C def)
{
    if (!obj.count(key)) {
        val = def;
        return;
    }
    try {
        jsonCast(obj[key], val);
    } catch (std::runtime_error &err) {
        throw std::runtime_error("While accessing key '" + key + "' -> "
                + err.what());
    }
}

/**
 * Join strings contained in \c strs and return resulting string.
 *
 * \c strs Vector of strings to be joined.
 * \ret String like "a, b, c"
 */
std::string join(const std::vector<std::string> &strs);

/*
 * Trim whitespace from word contained in \c word.
 *
 * Removes whitespace from the begining and end of \c word.
 */
std::string trim(const std::string &s);

/*
 * Trim whitespace from word contained in \c word.
 *
 * Removes whitespace from the begining and end of \c word.
 */
std::string trim(const std::string &s);

/*
 * Set thread name
 * \note: In HTOP go to Setup->Display Options and set 'Show custom thread
 * names'
 */
void set_thread_name(std::string name);

/**
 * ANSI escape character macro, for some color.
 */
#define ESC_CHR(CHR) (char) 27 << "[1;" << (int) CHR << "m"

#define ANSI_BOLD   1
#define ANSI_BLUE   34
#define ANSI_YELLOW 33
#define ANSI_GREEN  32
#define ANSI_RED    31
#define ANSI_RST    0
#endif // ifndef ARAXCONTROLLER_UTILITIES_HEADER_FILE
