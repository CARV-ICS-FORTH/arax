#include "utils/arax_plot.h"
#include "testing.h"

TEST_CASE("arax_plot tests")
{
    test_common_setup();

    int fd = test_open_config();

    const char *config = test_create_config(0x1000000);

    write(fd, config, strlen(config) );

    close(fd);

    arax_pipe_s *vpipe = arax_first_init();

    arax_controller_init_done();

    SECTION("test_with_null")
    {
        REQUIRE(arax_plot_register_metric("test", 0));
    }
    SECTION("test_with_shm_ptr")
    {
        uint64_t *ptr = (uint64_t *) arch_alloc_allocate(&(vpipe->allocator), sizeof(uint64_t));

        REQUIRE(arax_plot_register_metric("test", ptr));
    }

    /**
     * This test can not be potred to Catch2 as it has to abort(signal 6).
     * Catch2 at the time of writing does not support signals.
     * SECTION("test_with_bad_ptr")
     * {
     *  uint64_t *ptr = (uint64_t*)malloc(sizeof(uint64_t));
     *  arax_plot_register_metric("test", ptr);
     * }
     */
    arax_final_exit(vpipe);

    test_common_teardown();
}
