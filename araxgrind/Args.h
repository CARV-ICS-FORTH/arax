#include <string>
#include <set>

bool parseArgs(std::ostream & os, int argc, char *argv[]);

void printArgsHelp(std::ostream & os);

bool getHelp();

bool getAll();

bool getPtr();

bool getRefresh();

std::set<void *> getInspectPointers();

bool getList();

bool getTrack();
