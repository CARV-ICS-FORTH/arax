#include "Views.h"

#define TYPE_SIZE(TYPE) \
    ID_OUT << _TR(_TH(#TYPE) + _TD(_S(sizeof(TYPE)) + " B")) << std::endl

void viewStructSizes(std::ostream & out, std::string & id_str, int digits)
{
    ID_OUT << "<h2 onClick=\"blockTogle('size_block')\">Struct Sizes</h2>\n";
    ID_OUT << "<div class=block name=size_block>\n";
    ID_INC;
    ID_OUT << "<table>\n";
    ID_INC;
    ID_OUT << _TR(_TH("Type") + _TH("Size")) << std::endl;

    TYPE_SIZE(arax_proc_s);
    TYPE_SIZE(arax_accel_s);
    TYPE_SIZE(arax_data_s);
    TYPE_SIZE(arax_task_msg_s);
    TYPE_SIZE(arax_pipe_s);
    TYPE_SIZE(utils_queue_s);
    TYPE_SIZE(arax_vaccel_s);

    ID_DEC;
    ID_OUT << "</table>\n";
    ID_DEC;
    ID_OUT << "</div>\n";
}
