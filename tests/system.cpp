#include "utils/system.h"
#include "testing.h"

TEST_CASE("system tests")
{
    test_common_setup("");

    SECTION("test_exec_name")
    {
        std::string exec_name      = system_exec_name();
        std::string should_be_name = "system_unit";
        std::string binary_name    = exec_name.substr(exec_name.length() - should_be_name.length());

        REQUIRE(binary_name == should_be_name);
    }
    SECTION("test_thread_id")
    {
        REQUIRE(system_thread_id() > 0);
    }

    SECTION("test_backtrace")
    {
        system_backtrace(0);
    }

    test_common_teardown();
}
