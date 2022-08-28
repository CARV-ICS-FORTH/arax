#include "utils/config.h"
#include "testing.h"

TEST_CASE("Config tests")
{
    #define TEST_KEYS 7

    const char *vtalk_keys[TEST_KEYS] = {
        "test1", "test2", "test3", "test4", "test5", "test6", "test7"
    };
    const char *vtalk_vals[TEST_KEYS] = {
        "0", "1", "test3", "4096", "-4096", "0xFFFFFFFF", "0xFFFFFFFFF"
    };

    int cnt;
    int fd;

    test_common_setup();

    fd = test_open_config();
    for (cnt = 0; cnt < TEST_KEYS; cnt++) {
        write(fd, vtalk_keys[cnt], strlen(vtalk_keys[cnt]) );
        write(fd, " ", 1);
        write(fd, vtalk_vals[cnt], strlen(vtalk_vals[cnt]) );
        write(fd, "\n", 1);
    }
    close(fd);

    DYNAMIC_SECTION("test_config_no_file")
    {
        char *conf_file = utils_config_alloc_path(ARAX_CONFIG_FILE);
        int temp;

        REQUIRE_FALSE(unlink(conf_file) ); /* Remove test file*/

        REQUIRE_FALSE(utils_config_get_int(conf_file, "SHOULD_FAIL", &temp, 0) );

        utils_config_free_path(conf_file);
        close(test_open_config() ); /* Recreate it for teardown */
    }

    DYNAMIC_SECTION("test_config_path_alloc")
    {
        char temp[4096];
        char *path;

        path = utils_config_alloc_path("~");
        REQUIRE(std::string(path) == std::string(system_home_path()));
        utils_config_free_path(path);

        path = utils_config_alloc_path("@~@");
        sprintf(temp, "@%s@", system_home_path());
        REQUIRE(std::string(path) == std::string(temp));
        utils_config_free_path(path);

        path = utils_config_alloc_path("~~");
        sprintf(temp, "%s%s", system_home_path(), system_home_path());
        REQUIRE(std::string(path) == std::string(temp));
        utils_config_free_path(path);

        REQUIRE(utils_config_alloc_path(0) == 0);
    }

    for (int test = 0; test < TEST_KEYS; test++) {
        DYNAMIC_SECTION("test_config_get_str " << vtalk_keys[test])
        {
            char temp[32];
            char *conf = utils_config_alloc_path(ARAX_CONFIG_FILE);

            REQUIRE(utils_config_get_str(conf, vtalk_keys[test], temp, 32, 0) );
            REQUIRE(std::string(temp) == vtalk_vals[test]);

            utils_config_free_path(conf);
        }

        DYNAMIC_SECTION("test_config_get_str_fail" << vtalk_keys[test])
        {
            char temp[32];
            int tret[TEST_KEYS] = {
                0, 0, 1, 0, 0, 0
            };

            char *conf = utils_config_alloc_path(ARAX_CONFIG_FILE);

            REQUIRE(!!utils_config_get_str(conf, vtalk_vals[test], temp, 32, "FAIL") == tret[test]);

            utils_config_free_path(conf);
            if (tret[test])
                REQUIRE(std::string(temp) == vtalk_vals[test]);
            else
                REQUIRE(std::string(temp) == "FAIL");
        }

        DYNAMIC_SECTION("test_config_get_bool" << vtalk_keys[test])
        {
            int temp;
            char *conf = utils_config_alloc_path(ARAX_CONFIG_FILE);
            int tvals[TEST_KEYS] = {
                0, 1, 0, 0, 0, 0
            };
            int tret[TEST_KEYS] = {
                1, 1, 0, 0, 0, 0
            };

            REQUIRE(utils_config_get_bool(conf, vtalk_keys[test], &temp,
              !tvals[test]) == tret[test]);
            if (tret[test])
                REQUIRE(temp == tvals[test]);
            else
                REQUIRE(temp == !tvals[test]);

            utils_config_free_path(conf);
        }

        DYNAMIC_SECTION("test_config_get_int" << vtalk_keys[test])
        {
            int temp;
            char *conf = utils_config_alloc_path(ARAX_CONFIG_FILE);
            long tvals[TEST_KEYS] = {
                0, 1, 0, 4096, -4096, 0xFFFFFFFF, 0
            };
            int tret[TEST_KEYS] = {
                1, 1, 0, 1, 1, 0
            };

            REQUIRE(utils_config_get_int(conf, vtalk_keys[test], &temp,
              !tvals[test]) == tret[test]);
            if (tret[test])
                REQUIRE(temp == tvals[test]);
            else
                REQUIRE(temp == !tvals[test]);

            utils_config_free_path(conf);
        }
    }

    test_common_teardown();
}
