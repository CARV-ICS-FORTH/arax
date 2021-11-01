#include "utils/Kv.h"
#include "testing.h"
#define TEST_LENGTH VINE_KV_CAP

TEST_CASE("kv_tests")
{
    utils_kv_s kv;

    test_common_setup();
    utils_kv_init(&kv);
    REQUIRE(kv.pairs == 0);

    SECTION("test_get")
    {
        REQUIRE(utils_kv_get(&kv, 0) == 0);
    }


    SECTION("test_set")
    {
        REQUIRE(utils_kv_get(&kv, 0) == 0);
        utils_kv_set(&kv, 0, (void *) 1);
        REQUIRE(utils_kv_get(&kv, 0));
        REQUIRE(*utils_kv_get(&kv, 0) == (void *) 1);
        utils_kv_set(&kv, 0, (void *) 2);
        REQUIRE(utils_kv_get(&kv, 0));
        REQUIRE(*utils_kv_get(&kv, 0) == (void *) 2);
        utils_kv_set(&kv, 0, (void *) 0);
        REQUIRE(utils_kv_get(&kv, 0));
        REQUIRE(*utils_kv_get(&kv, 0) == (void *) 0);
    }


    SECTION("test_get_set")
    {
        size_t cnt;

        for (cnt = 0; cnt < TEST_LENGTH; cnt++) {
            REQUIRE(utils_kv_get(&kv, (void *) cnt) == 0);
            utils_kv_set(&kv, (void *) cnt, (void *) cnt);
            REQUIRE(*utils_kv_get(&kv, (void *) cnt) == (void *) cnt);
        }

        for (cnt = 0; cnt < TEST_LENGTH; cnt++) {
            REQUIRE(*utils_kv_get(&kv, (void *) cnt) == (void *) cnt);
        }
    }

    test_common_teardown();
}
