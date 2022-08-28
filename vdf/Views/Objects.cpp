#include "Views.h"

std::string procTypes(arax_proc_s *proc)
{
    std::string str;
    std::string sep = "";

    for (int t = 1; t < ARAX_ACCEL_TYPES; t++) {
        arax_accel_type_e type = (arax_accel_type_e) t;
        if (arax_proc_get_functor(proc, type)) {
            str += sep + arax_accel_type_to_str(type);
            sep  = ", ";
        }
    }
    return _TD(str);
}

void viewObjects(std::ostream & out, std::string & id_str, int digits)
{
    int type;
    utils_list_s *list;
    utils_list_node_s *itr;
    arax_object_s *obj;

    ID_OUT << "<h2 onClick=\"blockTogle('obj_block')\">Objects</h2>\n";
    ID_OUT << "<div class=block name=obj_block>\n";
    ID_INC;

    ID_OUT << "<div class=hgroup>\n";
    ID_INC;
    for (type = 0; type < ARAX_TYPE_COUNT; type++) {
        list = arax_object_list_lock(&(vpipe->objs), (arax_object_type_e) type);
        ID_OUT << "<div class='bg" << type % 2 << "'>\n";
        ID_INC;
        ID_OUT << "<table>\n";
        ID_INC;
        ID_OUT <<
            _TR(_TH(std::string(arax_object_type_to_str((arax_object_type_e) type)) + "[" + _S(list->length) + "]",
          "colspan=5")) << std::endl;
        ID_OUT << _TR(_TH("Address") + _TH("Name") + _TH("Refs") + _TH("Type") + _TH("Extra")) << std::endl;
        if (list->length) {
            utils_list_for_each(*list, itr)
            {
                obj = (arax_object_s *) itr->owner;
                ID_OUT << "<tr onmouseover=\"highlight_same(this)\" name=\"alloc"
                       << minPtr(obj,
                  digits) << "\"><th>"
                       << minPtr(obj, digits) << "</th>" << _TD(obj->name) << _TD(_S(arax_object_refs(obj)));
                switch (type) {
                    case ARAX_TYPE_PHYS_ACCEL:
                        ID_OUT << getAcceleratorType((arax_accel_s *) obj)
                               << _TD("Rev:" + _S(arax_accel_get_revision(((arax_accel_s *) obj))));
                        break;
                    case ARAX_TYPE_VIRT_ACCEL:
                        ID_OUT << getAcceleratorType((arax_vaccel_s *) obj)
                               << _TD("Queue:" + _S(utils_queue_used_slots(arax_vaccel_queue(
                              (arax_vaccel_s *) obj))));
                        break;
                    case ARAX_TYPE_PROC:
                        ID_OUT << procTypes((arax_proc_s *) obj) << _TD("");
                        break;
                    case ARAX_TYPE_DATA:
                        ID_OUT << _TD(_S(((arax_data_s *) obj)->size)) << _TD("");
                        break;
                    case ARAX_TYPE_TASK: {
                        arax_task_msg_s *task = (arax_task_msg_s *) obj;
                        ID_OUT << getAcceleratorType(((arax_accel_s *) ((task)->accel)))
                               << _TD(((arax_object_s *) ((task)->proc))->name);
                    }
                    break;
                    default:
                        ID_OUT << _TD("Unknown") << _TD("");
                        break;
                }
                ID_OUT << "</tr>\n";
            }
        } else {
            ID_OUT <<
                _TR(_TH(std::string("No ") + arax_object_type_to_str((arax_object_type_e) type),
              "colspan=5")) << std::endl;
        }
        ID_DEC;
        ID_OUT << "</table>\n";
        arax_object_list_unlock(&(vpipe->objs), (arax_object_type_e) type);
        ID_DEC;
        ID_OUT << "</div>\n";
    }
    ID_DEC;
    ID_OUT << "</div>\n";
    ID_DEC;
    ID_OUT << "</div>\n";
} // viewObjects
