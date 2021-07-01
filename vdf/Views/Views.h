#ifndef VDF_VIEWS_HEADER
#define VDF_VIEWS_HEADER

#include <iostream>
#include <vine_pipe.h>
#include "core/vine_data.h"
#include "../Misc.h"

#define ID_OUT out << id_str
#define ID_INC id_str += '\t'
#define ID_DEC id_str.resize(id_str.size() - 1)

void viewAllocations(std::ostream & out, std::string & id_str, int digits);
void viewConfig(std::ostream & out, std::string & id_str, int digits);
void viewObjects(std::ostream & out, std::string & id_str, int digits);
void viewStructSizes(std::ostream & out, std::string & id_str, int digits);
void viewThrottles(std::ostream & out, std::string & id_str, int digits);
template <class T>
std::string getAcceleratorType(T *obj)
{
    return _TD(vine_accel_type_to_str(obj->type));
}

extern vine_pipe_s *vpipe;
#endif // ifndef VDF_VIEWS_HEADER
