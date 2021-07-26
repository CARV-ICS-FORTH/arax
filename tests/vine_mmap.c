#include "testing.h"
#include <sys/mman.h>
#include "utils/bitmap.h"

void setup()
{
    test_common_setup();

    int fd = test_open_config();

    const char *config = test_create_config(0xa00000);

    write(fd, config, strlen(config) );

    close(fd);
}

void teardown()
{
    test_common_teardown();
}

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

START_TEST(mmap_ummap)
{
    vine_pipe_s *vpipe = vine_first_init();

    srand(0);

    for (int t = 1; t < 1000; t++) {
        size_t s  = alloc_size();
        void *ptr = vine_mmap(s);
        ck_assert(utils_bitmap_count_allocated(arch_alloc_get_bitmap()) == utils_bitmap_used(arch_alloc_get_bitmap()));
        vine_ummap(ptr, s);
        ck_assert(utils_bitmap_count_allocated(arch_alloc_get_bitmap()) == utils_bitmap_used(arch_alloc_get_bitmap()));
    }

    vine_final_exit(vpipe);
}
END_TEST

Suite* suite_init()
{
    Suite *s;
    TCase *tc_single;

    s         = suite_create("vine_mmap");
    tc_single = tcase_create("Single");
    tcase_add_unchecked_fixture(tc_single, setup, teardown);
    tcase_add_test(tc_single, mmap_ummap);
    suite_add_tcase(s, tc_single);
    return s;
}

int main(int argc, char *argv[])
{
    Suite *s;
    SRunner *sr;
    int failed;

    s  = suite_init();
    sr = srunner_create(s);
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_NORMAL);
    failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
