#include "Views.h"
#include <fstream>
#include <Poco/Path.h>

void viewConfig(std::ostream & out, std::string & id_str, int digits)
{
    ID_OUT << "<h2 onClick=\"blockTogle('conf_block')\">Config</h2>\n";
    ID_OUT << "<div class=block name=conf_block >\n";
    ID_INC;
    ID_OUT << "<table>\n";
    ID_INC;
    ID_OUT << _TR(_TH("Key") + _TH("Value")) << std::endl;


    std::ifstream cfg(Poco::Path::expand(VINE_CONFIG_FILE));

    if (!cfg)
        ID_OUT << _TR(_TH("File") + _TD(Poco::Path::expand(VINE_CONFIG_FILE) + "(NotFound!)")) << std::endl;
    else
        ID_OUT << _TR(_TH("File") + _TD(Poco::Path::expand(VINE_CONFIG_FILE))) << std::endl;

    std::ostringstream iss;

    iss << (void *) vpipe;

    ID_OUT << _TR(_TH("Base") + _TD(iss.str())) << std::endl;
    ID_OUT << _TR(normalize("Size", vpipe->shm_size)) << std::endl;

    ID_OUT << _TR(_TH("") + _TH("")) << std::endl;
    do{
        std::string key, value;
        cfg >> key >> value;
        if (cfg)
            ID_OUT << _TR(_TH(key) + _TD(value)) << std::endl;
    }while(cfg);

    ID_DEC;
    ID_OUT << "</table>\n";
    ID_DEC;
    ID_OUT << "</div>\n";
} // viewConfig
