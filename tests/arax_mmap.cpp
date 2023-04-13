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

void* arax_mmap(size_t s);
void* arax_ummap(void *a, size_t s);
utils_bitmap_s* arch_alloc_get_bitmap();

TEST_CASE("arax_mmap/arax_ummap test")
{
    const char *config = test_create_config(0xa00000);

    test_common_setup(config);

    arax_pipe_s *vpipe = arax_first_init();

    srand(0);

    void *test_mmap = arax_mmap(4096);

    if (!test_mmap)
        goto SKIP_TEST;

    arax_ummap(test_mmap, 4096);

    for (int t = 1; t < 1000; t++) {
        size_t s = alloc_size();
        DYNAMIC_SECTION("mmap_ummap Size:" << s)
        {
            void *ptr = arax_mmap(s);

            REQUIRE(utils_bitmap_count_allocated(arch_alloc_get_bitmap()) ==
              utils_bitmap_used(arch_alloc_get_bitmap()));
            arax_ummap(ptr, s);
            REQUIRE(utils_bitmap_count_allocated(arch_alloc_get_bitmap()) ==
              utils_bitmap_used(arch_alloc_get_bitmap()));
        }
    }


SKIP_TEST:

    arax_final_exit(vpipe);

    test_common_teardown();
}
