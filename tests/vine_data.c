#include "testing.h"
#include "vine_pipe.h"
#include "core/vine_data.h"

void setup()
{
    test_common_setup();

    int fd = test_open_config();

    const char *config = test_create_config(0x1000000);

    write(fd, config, strlen(config) );

    close(fd);
}

void teardown()
{
    test_common_teardown();
}

START_TEST(test_data_leak)
{
    fprintf(stderr, "Hello %d", __LINE__);
    vine_pipe_s *vpipe = vine_first_init();
    // vine_data_deref causes buffer allocation in shm, ensure throttle aggrees
    size_t initial_space = vine_throttle_get_available_size(&(vpipe->throttle));
    size_t capacity      = vine_throttle_get_total_size(&(vpipe->throttle));

    ck_assert_uint_ge(capacity, initial_space); // Space should always be <= capacity

    vine_data_s *data = vine_data_init(vpipe, 0, 1);

    vine_data_free(data);

    size_t final_space = vine_throttle_get_available_size(&(vpipe->throttle));

    ck_assert_uint_eq(initial_space, final_space); // Leak in metadata


    data = vine_data_init(vpipe, 0, 1);
    vine_data_deref(data);
    vine_data_free(data);

    final_space = vine_throttle_get_available_size(&(vpipe->throttle));

    ck_assert_uint_eq(initial_space, final_space); // Leak in metadata


    vine_final_exit(vpipe);
}
END_TEST START_TEST(test_alloc_data_alligned)
{
    fprintf(stderr, "Hello %d\n\n", __LINE__);
    vine_pipe_s *vpipe = vine_first_init();
    size_t size        = _i & 256;
    size_t align       = 1 << (_i / 256);

    vine_data *data = vine_data_init_aligned(vpipe, 0, size, align);

    vine_data_stat(data);

    vine_data_check_flags(data);

    ck_assert(data != NULL);

    ck_assert(vine_data_deref(data) != NULL);

    ck_assert((size_t) vine_data_deref(data) % align == 0);

    ck_assert_ptr_eq(vine_data_ref(vine_data_deref(data)), data);

    ck_assert_int_eq(vine_data_size(data), size);

    ck_assert(!vine_data_check_ready(vpipe, data));
    vine_data_mark_ready(vpipe, data);
    ck_assert(vine_data_check_ready(vpipe, data));

    // because dec of buffer take place on task_issue
    // i have to dec buffer otherwise there is a leak here
    #ifdef VINE_THROTTLE_DEBUG
    printf("%s\t", __func__);
    #endif
    vine_pipe_size_dec(vpipe, size + ((vine_data_s *) data)->align + sizeof(size_t *) );

    vine_data_free(data);

    vine_final_exit(vpipe);
} /* START_TEST */

END_TEST START_TEST(test_alloc_data)
{
    fprintf(stderr, "Hello %d\n\n", __LINE__);
    vine_pipe_s *vpipe = vine_first_init();
    size_t size        = _i;

    // Physical accel
    vine_accel_s *phys = vine_accel_init(vpipe, "FakePhysAccel", 0, 100, 10000);

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 1);

    // Virtual accels - assigned to phys
    vine_vaccel_s *vac_1 = vine_accel_acquire_type(ANY);

    ck_assert(vac_1);
    vine_accel_add_vaccel(phys, vac_1);
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL), 1);

    vine_vaccel_s *vac_2 = vine_accel_acquire_type(ANY);

    ck_assert(vac_2);
    vine_accel_add_vaccel(phys, vac_2);
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL), 2);

    vine_data *data = VINE_BUFFER(0, size);

    ck_assert_int_eq(vine_object_refs(data), 1);

    ck_assert(data != NULL);

    ck_assert(vine_data_has_remote(data) == 0);

    ck_assert_ptr_eq(vine_data_deref(data), ((vine_data_s *) data)->buffer);

    ck_assert(vine_data_deref(data) != NULL);

    ck_assert_ptr_eq(vine_data_ref(vine_data_deref(data)), data);

    vine_data_check_flags(data);

    ck_assert_int_eq(vine_data_size(data), size);

    ck_assert(!vine_data_check_ready(vpipe, data));
    vine_data_mark_ready(vpipe, data);
    ck_assert(vine_data_check_ready(vpipe, data));

    // Just call these functions - they should not crash
    // Eventually add more thorough tests.

    ck_assert_int_eq(vine_object_refs(data), 1);
    ck_assert_ptr_eq(0, ((vine_data_s *) data)->accel);
    vine_data_arg_init(data, vac_1);
    ck_assert_ptr_eq(vac_1, ((vine_data_s *) data)->accel);
    ck_assert_int_eq(vine_object_refs(data), 1);
    vine_data_input_init(data, vac_1);
    ck_assert_ptr_eq(vac_1, ((vine_data_s *) data)->accel);
    ck_assert_int_eq(vine_object_refs(data), 2);
    vine_data_output_init(data, vac_1);
    ck_assert_ptr_eq(vac_1, ((vine_data_s *) data)->accel);
    ck_assert_int_eq(vine_object_refs(data), 3);
    vine_data_output_done(data);
    ck_assert_ptr_eq(vac_1, ((vine_data_s *) data)->accel);
    ck_assert_int_eq(vine_object_refs(data), 3);
    vine_data_memcpy(0, data, data, 0);
    ck_assert_ptr_eq(vac_1, ((vine_data_s *) data)->accel);

    // Repeat tests , with different vac, but pointing to same phys

    ck_assert_int_eq(vine_object_refs(data), 3);
    ck_assert_ptr_eq(vac_1, ((vine_data_s *) data)->accel);
    vine_data_arg_init(data, vac_2);
    ck_assert_ptr_eq(vac_2, ((vine_data_s *) data)->accel);
    ck_assert_int_eq(vine_object_refs(data), 3);
    vine_data_input_init(data, vac_2);
    ck_assert_ptr_eq(vac_2, ((vine_data_s *) data)->accel);
    ck_assert_int_eq(vine_object_refs(data), 4);
    vine_data_output_init(data, vac_2);
    ck_assert_ptr_eq(vac_2, ((vine_data_s *) data)->accel);
    ck_assert_int_eq(vine_object_refs(data), 5);
    vine_data_output_done(data);
    ck_assert_int_eq(vine_object_refs(data), 5);
    vine_data_memcpy(0, data, data, 0);
    ck_assert_ptr_eq(vac_2, ((vine_data_s *) data)->accel);

    // vine_data_sync_to_remote should be a no-op when data are in remote

    vine_data_modified(data, REMT_SYNC);
    vine_data_sync_to_remote(vac_1, data, 0);

    // vine_data_sync_from_remote should be a no-op when data are in USR

    vine_data_modified(data, USER_SYNC);
    vine_data_sync_from_remote(vac_1, data, 0);

    // vine_data_sync_from_remote should be a no-op when data are in SHM (and user pointer is null)

    vine_data_modified(data, SHM_SYNC);
    vine_data_sync_from_remote(vac_1, data, 0);

    // because dec of buffer take place on task_issue
    // i have to dec buffer otherwise there is a leak here
    #ifdef VINE_THROTTLE_DEBUG
    printf("%s\t", __func__);
    #endif
    vine_pipe_size_dec(vpipe, size + ((vine_data_s *) data)->align + sizeof(size_t *) );

    unsigned int i = 5;
    ck_assert_int_eq(i, vine_object_refs(data)); // We did 5 vine_data_*_init calls
    for (; i > 0; i--)
        vine_data_free(data);

    vine_accel_release((vine_accel **) &vac_1);

    vine_accel_release((vine_accel **) &vac_2);

    vine_accel_release((vine_accel **) &phys);

    vine_final_exit(vpipe);
} /* START_TEST */

END_TEST START_TEST(test_data_ref_offset)
{
    fprintf(stderr, "Hello %d\n\n", __LINE__);
    vine_pipe_s *vpipe = vine_first_init();
    vine_data_s *data  = vine_data_init(vpipe, 0, 16);
    void *start        = vine_data_deref(data);
    void *end      = vine_data_deref(data) + vine_data_size(data);
    void *test_ptr = start + _i;

    if (test_ptr >= start && test_ptr < end) { // Should be inside buffer
        ck_assert_ptr_eq(vine_data_ref_offset(vpipe, test_ptr), data);
    } else { // 'Outside' of buffer range
        ck_assert_ptr_eq(vine_data_ref_offset(vpipe, test_ptr), 0);
    }

    vine_data_free(data);

    vine_final_exit(vpipe);
}

END_TEST


Suite* suite_init()
{
    Suite *s;
    TCase *tc_single;

    s         = suite_create("Vine Talk");
    tc_single = tcase_create("Single");
    tcase_add_unchecked_fixture(tc_single, setup, teardown);
    tcase_add_loop_test(tc_single, test_alloc_data, 0, 2);
    tcase_add_test(tc_single, test_data_leak);
    tcase_add_loop_test(tc_single, test_alloc_data_alligned, 1, 8);
    tcase_add_loop_test(tc_single, test_data_ref_offset, -24, 24);
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
    srunner_set_fork_status(sr, CK_FORK);
    srunner_run_all(sr, CK_NORMAL);
    failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
