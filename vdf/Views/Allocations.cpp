#include "Views.h"
#include <vector>

struct allocation
{
    void *      name;
    std::size_t start;
    std::size_t end;
    std::size_t used;
};

std::ostream & operator << (std::ostream & os, const struct allocation &alloc)
{
    int digits    = calcDigits(vpipe, vpipe->shm_size);
    int64_t space = alloc.end - alloc.start;
    std::string us;

    if (space == alloc.used)
        us = _TD(_S(space), "colspan=2");
    else
        us = _TD(_S(alloc.used)) + _TD(_S(space));
    os << "<tr onmouseover=\"highlight_same(this)\" name=\"alloc" << minPtr(alloc.name, digits) << "\">"
       << _TD(_S(alloc.start) + " - " + _S(alloc.end)) + us
       << std::endl;
    return os;
}

void inspector(void *start, void *end, std::size_t used, void *arg)
{
    std::vector<allocation> *alloc_vec = (std::vector<allocation> *) arg;

    if (used)
        used -= sizeof(std::size_t);
    allocation alloc = { start, (std::size_t) start, (std::size_t) end, used };

    alloc_vec->push_back(alloc);
}

void viewAllocations(std::ostream & out, std::string & id_str, int digits)
{
    arch_alloc_stats_s stats = arch_alloc_stats(&(vpipe->allocator));

    ID_OUT << "<h2 onClick=\"blockTogle('alloc_block')\">Allocations</h2>\n";
    ID_OUT << "<div class=block name=alloc_block>\n";
    ID_INC;
    ID_OUT << "<div class=vgroup>\n";
    ID_INC;
    ID_OUT << "<div class=hgroup>\n";
    ID_INC;
    ID_OUT << "<table>\n";
    ID_INC;
    ID_OUT << _TR(_TH("Fine Allocator", "colspan=2")) << std::endl;
    ID_OUT << "<tr><th>Base</th><td>" << vpipe << "</td></tr>\n";
    ID_OUT << _TR(normalize("Space", stats.total_bytes)) << std::endl;
    ID_OUT << _TR(normalize("Used", stats.used_bytes)) << std::endl;
    ID_OUT << _TR(normalize("Free", stats.total_bytes - stats.used_bytes)) << std::endl;
    #ifdef ALLOC_STATS
    ID_OUT << _TR(_TH("Failed allocations") + _TD(_S(stats.allocs[0]))) << std::endl;
    ID_OUT << _TR(_TH("Good allocations") + _TD(_S(stats.allocs[1]))) << std::endl;
    auto total_allocs = stats.allocs[0] + stats.allocs[1];
    ID_OUT << _TR(_TH("Total Alloc") + _TD(_S(total_allocs))) << std::endl;
    ID_OUT << _TR(_TH("Total Free") + _TD(_S(stats.frees))) << std::endl;
    auto leaks = stats.allocs[1] - stats.frees;
    ID_OUT << _TR(_TH("Leaks") + _TD(_S(leaks) + "(" + _S((leaks * 100) / total_allocs) + "&#37;)")) << std::endl;
    #endif
    ID_DEC;
    ID_OUT << "</table>\n";

    auto bmp = arch_alloc_get_bitmap();

    if (bmp) {
        ID_OUT << "<table>\n";
        ID_INC;
        ID_OUT << _TR(_TH("Coarse Allocator", "colspan=2")) << std::endl;
        ID_OUT << _TR(_TH("Space") + _TD(_S(utils_bitmap_size(bmp)))) << std::endl;
        ID_OUT << _TR(_TH("Used") + _TD(_S(utils_bitmap_used(bmp)))) << std::endl;
        ID_OUT << _TR(_TH("Free") + _TD(_S(utils_bitmap_free(bmp)))) << std::endl;
        ID_DEC;
        ID_OUT << "</table>\n";
    }
    ID_DEC;
    ID_OUT << "</div>\n";

    std::vector<allocation> allocs;

    #ifdef ALLOC_STATS
    allocs.reserve(stats.allocs[1]); // Optional Optimization
    #endif

    arch_alloc_inspect(&(vpipe->allocator), inspector, &allocs);

    std::size_t base = (std::size_t) ((&(vpipe->allocator)) + 1);

    for (auto alloc : allocs) {
        alloc.start -= base;
        alloc.end   -= base;
    }
    allocs.clear();

    ID_DEC;
    ID_OUT << "</div>\n";
    ID_DEC;
    ID_OUT << "</div>\n";
} // viewAllocations
