#include "testing.h"
#include <sys/mman.h>
#include "utils/bitmap.h"

size_t alloc_size()
{
    if (rand() % 2)
        return 4096 * (1 + rand() % 32);
    else
        return ((rand() % 16) + 1) * 256 * 1024;
}

void* vine_mmap(size_t s);
void* vine_ummap(void *a, size_t s);
utils_bitmap_s* arch_alloc_get_bitmap();

TEST_CASE("vine_mmap/vine_ummap test")
{
    test_common_setup();

    int fd = test_open_config();

    const char *config = test_create_config(0xa00000);

    write(fd, config, strlen(config) );

    close(fd);

    vine_pipe_s *vpipe = vine_first_init();

    srand(0);

    for (int t = 1; t < 1000; t++) {
        size_t s = alloc_size();
        DYNAMIC_SECTION("mmap_ummap Size:" << s)
        {
            void *ptr = vine_mmap(s);

            REQUIRE(utils_bitmap_count_allocated(arch_alloc_get_bitmap()) ==
              utils_bitmap_used(arch_alloc_get_bitmap()));
            vine_ummap(ptr, s);
            REQUIRE(utils_bitmap_count_allocated(arch_alloc_get_bitmap()) ==
              utils_bitmap_used(arch_alloc_get_bitmap()));
        }
    }


    vine_final_exit(vpipe);

    test_common_teardown();
}
