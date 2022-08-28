#include "Views.h"

void printThrotle(std::ostream & out, std::string & id_str, arax_throttle_s *th, std::string name)
{
    std::size_t a = arax_throttle_get_available_size(th);
    std::size_t t = arax_throttle_get_total_size(th);

    ID_OUT << "<table>\n";
    ID_INC;
    ID_OUT << "<tr><th>" << name << "</th></tr>\n";
    ID_OUT << "<tr><td class='nopad'><div class='throt_bar' style='width:" << ((t - a) * 100) / t
           << "%'></div></td></tr>\n";
    ID_OUT << "<tr><td>" << autoBytes(t - a) << '/' << autoBytes(t) << "</td></tr>\n";
    ID_DEC;
    ID_OUT << "</table>\n";
}

void viewThrottles(std::ostream & out, std::string & id_str, int digits)
{
    utils_list_s *list;
    utils_list_node_s *itr;

    ID_OUT << "<h2 onClick=\"blockTogle('throt_block')\">Throttling</h2>\n";
    ID_OUT << "<div class=block name=throt_block>\n";
    ID_INC;
    ID_OUT << "<div>";
    list = arax_object_list_lock(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL);
    std::size_t p_cnt = list->length;

    if (p_cnt) {
        ID_INC;
        ID_OUT << "<div class='hgroup'>";
        utils_list_for_each(*list, itr)
        {
            auto p = (arax_accel_s *) itr->owner;

            printThrotle(out, id_str, &(p->throttle), p->obj.name);
        }
        ID_DEC;
        ID_OUT << "</div>\n";
    }
    arax_object_list_unlock(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL);
    ID_INC;
    ID_OUT << "<div class='hgroup'>";
    printThrotle(out, id_str, &(vpipe->throttle), "Shm");
    ID_DEC;
    ID_OUT << "</div>\n";
    ID_DEC;
    ID_OUT << "</div>\n";
    ID_DEC;
    ID_OUT << "</div>\n";
} // viewThrottles
