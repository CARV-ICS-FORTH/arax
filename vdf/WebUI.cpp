#include "WebUI.h"
#include <conf.h>
#include <Poco/URI.h>
#include <sstream>
#include <fstream>
#include <random>
#include "Pallete.h"
#include "arch/alloc.h"
#include "Views/Views.h"
using namespace Poco;
using namespace Poco::Net;

int bar_count = 0;

char hostname[1024];

void WebUI :: handleRequest(HTTPServerRequest & request, HTTPServerResponse & response)
{
    std::string id_str = "";

    response.setStatus(HTTPResponse::HTTP_OK);
    response.setContentType("text/html");
    URI uri(request.getURI());

    std::ostream& out = response.send();

    int digits = calcDigits(vpipe, vpipe->shm_size);

    if (!args["embed"]) {
        ID_OUT << "<!DOCTYPE html>\n";
        ID_OUT << "<html>\n";
        ID_INC;
        ID_OUT << "<head>\n";
        ID_INC;
        ID_OUT << "<title>VineWatch</title>\n";
        ID_OUT << "<link href=\"https://fonts.googleapis.com/css?family=Roboto\" rel=\"stylesheet\">\n";
        ID_DEC;
        ID_OUT << "</head>\n";
        ID_OUT << "<body>\n";
        ID_INC;
    }

    std::string src_path = __FILE__;

    src_path.resize(src_path.size() - 9);

    ID_OUT << "<style>";
    ID_OUT << "\n" << std::ifstream(src_path + "style.css").rdbuf();
    ID_OUT << "</style>\n";


    ID_OUT << "<script>";
    ID_OUT << "\n" << std::ifstream(src_path + "script.js").rdbuf();
    ID_OUT << "</script>\n";

    ID_OUT << "<div class=version>" << VINE_TALK_GIT_REV << " - " << VINE_TALK_GIT_BRANCH << "</div>\n";

    ID_OUT << std::ifstream(src_path + "logo.svg").rdbuf();

    if (!args["noconf"])
        viewConfig(out, id_str, digits);

    if (!args["nosizes"])
        viewStructSizes(out, id_str, digits);


    if (!args["nothrot"])
        viewThrottles(out, id_str, digits);

    if (!args["noalloc"])
        viewAllocations(out, id_str, digits);


    if (!args["noobj"])
        viewObjects(out, id_str, digits);

    if (!args["embed"]) {
        ID_DEC;
        ID_OUT << "</body>\n";
        ID_DEC;
        ID_OUT << "</html>\n";
    }
    out.flush();
} // WebUI::handleRequest

WebUI :: WebUI(std::map<std::string, bool> & args)
    : args(args)
{
    gethostname(hostname, 1024);
}
