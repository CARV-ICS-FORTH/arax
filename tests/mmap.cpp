#include "testing.h"
#include <sys/mman.h>

TEST_CASE("mmap behaviour")
{
    SECTION("forced")
    {
        unsigned int *map1 = (unsigned int *) mmap(0, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED | MAP_ANON,
            -1, 0);

        REQUIRE(map1 != MAP_FAILED);
        memset(map1, 0xFE, 4096);
        REQUIRE(*map1 == 0xFEFEFEFE);
        unsigned int *map2 = (unsigned int *) mmap(map1, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
            MAP_SHARED | MAP_ANON | MAP_FIXED, -1, 0);

        REQUIRE(map2 != MAP_FAILED);
        REQUIRE(*map2 != 0xFE);
        REQUIRE(*map2 == 0);
    }

    SECTION("lax")
    {
        unsigned int *map1 = (unsigned int *) mmap(0, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED | MAP_ANON,
            -1, 0);

        REQUIRE(map1 != MAP_FAILED);
        memset(map1, 0xFE, 4096);
        REQUIRE(*map1 == 0xFEFEFEFE);
        REQUIRE(*map1 != 0);

        unsigned int *map2 = (unsigned int *) mmap(map1, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
            MAP_SHARED | MAP_ANON, -1, 0);

        REQUIRE(map2 != MAP_FAILED);
        REQUIRE(map2 != map1);
        REQUIRE(*map2 != 0xFEFEFEFE);
        REQUIRE(*map2 == 0);
        REQUIRE(*map1 == 0xFEFEFEFE);
        REQUIRE(*map1 != 0);
    }
}
