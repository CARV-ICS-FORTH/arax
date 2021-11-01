#include "testing.h"
#include "struct_interop.h"

TEST_CASE("Struct interoperability test")
{
    SECTION("Test")
    {
        size_t c_sizes[STRUCT_INTEROP_SIZES];
        size_t cpp_sizes[STRUCT_INTEROP_SIZES];

        get_c_sizes(c_sizes);
        get_cpp_sizes(cpp_sizes);

        for (int cnt = 0; cnt < STRUCT_INTEROP_SIZES; cnt++) {
            fprintf(stderr, "Size[%d]: %lu %lu\n", cnt, c_sizes[cnt], cpp_sizes[cnt]);
            REQUIRE(c_sizes[cnt] == cpp_sizes[cnt]);
        }
    }
}
