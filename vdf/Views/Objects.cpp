#include "Views.h"

std::string procTypes(vine_proc_s *proc)
{
    std::string str;
    std::string sep = "";

    for (int t = 1; t < VINE_ACCEL_TYPES; t++) {
        vine_accel_type_e type = (vine_accel_type_e) t;
        if (vine_proc_get_functor(proc, type)) {
            str += sep + vine_accel_type_to_str(type);
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
    vine_object_s *obj;

    ID_OUT << "<h2 onClick=\"blockTogle('obj_block')\">Objects</h2>\n";
    ID_OUT << "<div class=block name=obj_block>\n";
    ID_INC;

    const char *typestr[VINE_TYPE_COUNT] =
    {
        "Phys Accel",
        "Virt Accel",
        "Vine Procs",
        "Vine Datas",
        "Vine Tasks"
    };

    ID_OUT << "<div class=hgroup>\n";
    ID_INC;
    for (type = 0; type < VINE_TYPE_COUNT; type++) {
        list = vine_object_list_lock(&(vpipe->objs), (vine_object_type_e) type);
        ID_OUT << "<div class='bg" << type % 2 << "'>\n";
        ID_INC;
        ID_OUT << "<table>\n";
        ID_INC;
        ID_OUT << _TR(_TH(std::string(typestr[type]) + "[" + _S(list->length) + "]", "colspan=5")) << std::endl;
        ID_OUT << _TR(_TH("Address") + _TH("Name") + _TH("Refs") + _TH("Type") + _TH("Extra")) << std::endl;
        if (list->length) {
            utils_list_for_each(*list, itr)
            {
                obj = (vine_object_s *) itr->owner;
                ID_OUT << "<tr onmouseover=\"highlight_same(this)\" name=\"alloc"
                       << minPtr(obj,
                  digits) << "\"><th>"
                       << minPtr(obj, digits) << "</th>" << _TD(obj->name) << _TD(_S(vine_object_refs(obj)));
                switch (type) {
                    case VINE_TYPE_PHYS_ACCEL:
                        ID_OUT << getAcceleratorType((vine_accel_s *) obj)
                               << _TD("Rev:" + _S(vine_accel_get_revision(((vine_accel_s *) obj))));
                        break;
                    case VINE_TYPE_VIRT_ACCEL:
                        ID_OUT << getAcceleratorType((vine_vaccel_s *) obj)
                               << _TD("Queue:" + _S(utils_queue_used_slots(vine_vaccel_queue(
                              (vine_vaccel_s *) obj))));
                        break;
                    case VINE_TYPE_PROC:
                        ID_OUT << procTypes((vine_proc_s *) obj) << _TD("");
                        break;
                    case VINE_TYPE_DATA:
                        ID_OUT << _TD(_S(((vine_data_s *) obj)->size)) << _TD("");
                        break;
                    case VINE_TYPE_TASK: {
                        vine_task_msg_s *task = (vine_task_msg_s *) obj;
                        ID_OUT << getAcceleratorType(((vine_accel_s *) ((task)->accel)))
                               << _TD(((vine_object_s *) ((task)->proc))->name);
                    }
                    break;
                    default:
                        ID_OUT << _TD("Unknown") << _TD("");
                        break;
                }
                ID_OUT << "</tr>\n";
            }
        } else {
            ID_OUT << _TR(_TH(std::string("No ") + typestr[type], "colspan=5")) << std::endl;
        }
        ID_DEC;
        ID_OUT << "</table>\n";
        vine_object_list_unlock(&(vpipe->objs), (vine_object_type_e) type);
        ID_DEC;
        ID_OUT << "</div>\n";
    }
    ID_DEC;
    ID_OUT << "</div>\n";
    ID_DEC;
    ID_OUT << "</div>\n";
} // viewObjects
