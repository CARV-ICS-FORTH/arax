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

START_TEST(test_in_out){
    vine_pipe_s *vpipe  = vine_first_init();
    vine_pipe_s *vpipe2 = vine_talk_init();

    ck_assert(!!vpipe2);
    ck_assert_ptr_eq(vpipe, vpipe2);

    ck_assert_ptr_eq(vine_pipe_mmap_address(vpipe), vpipe);
    ck_assert_int_eq(vine_pipe_have_to_mmap(vpipe, system_process_id()), 0);
    vine_talk_exit();
    ck_assert_ptr_eq(vine_pipe_mmap_address(vpipe), vpipe);
    ck_assert_int_eq(vine_pipe_have_to_mmap(vpipe, system_process_id()), 0);
    vine_final_exit(vpipe);
}
END_TEST START_TEST(test_revision)
{
    char *rev;
    vine_pipe_s *vpipe = vine_first_init();

    /**
     * Break sha intentionally
     */
    rev = (char *) vine_pipe_get_revision(vpipe);
    ck_assert(!!rev);
    ck_assert_str_eq(rev, VINE_TALK_GIT_REV);
    vine_final_exit(vpipe);
}

END_TEST START_TEST(test_single_proc)
{
    vine_proc_s *proc;
    vine_pipe_s *vpipe = vine_first_init();
    char pd[] = "TEST_FUNCTOR";
    int cnt;

    proc = create_proc(vpipe, "TEST_PROC");

    ck_assert_ptr_eq(vine_proc_register("TEST_PROC"), proc);

    ck_assert_int_eq(vine_proc_can_run_at(proc, _i), 0);

    vine_proc_set_functor(proc, _i, (VineFunctor *) pd);

    ck_assert_int_eq(vine_proc_can_run_at(proc, _i), 1);

    VineFunctor *vf = vine_proc_get_functor(proc, _i);

    ck_assert_ptr_eq(vf, pd);

    for (cnt = !ANY; cnt < VINE_ACCEL_TYPES; cnt++) {
        if (cnt == _i) {
            ck_assert(vine_proc_get_functor(proc, cnt) != NULL);
        } else {
            ck_assert(!vine_proc_get_functor(proc, cnt) );
        }
        ck_assert_int_eq(vine_proc_can_run_at(proc, cnt), cnt == _i);
    }

    ck_assert(!vine_proc_put(proc) );

    ck_assert(!vine_proc_get("TEST_PROC") );

    vine_final_exit(vpipe);
} /* START_TEST */

END_TEST START_TEST(test_task_issue_and_wait_v1)
{
    vine_pipe_s *vpipe = vine_first_init();
    vine_vaccel_s *accel;
    vine_accel_type_e at = _i % VINE_ACCEL_TYPES;

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PROC), 0);

    vine_proc_s *issue_proc = create_proc(vpipe, "issue_proc");

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PROC), 1);

    accel = vine_accel_acquire_type(at);

    ck_assert_int_eq(vine_object_refs((vine_object_s *) accel), 1);

    void *task_handler_state = handle_n_tasks(UTILS_QUEUE_CAPACITY * 2, at);

    for (int cnt = 0; cnt < UTILS_QUEUE_CAPACITY * 2; cnt++) {
        vine_task *task = vine_task_issue(accel, issue_proc, 0, 0, 0, 0, 0, 0);

        ck_assert_int_eq(vine_object_refs((vine_object_s *) accel), 2);

        vine_task_wait_done(task);

        ck_assert_int_eq(vine_task_stat(task, 0), task_completed);

        vine_task_free(task);

        ck_assert_int_eq(vine_object_refs((vine_object_s *) accel), 1);
    }

    // Normally scheduler would set phys to something valid.
    ck_assert(accel->phys);

    vine_accel_s *phys = accel->phys;

    // The physical accelerator should only be referenced by accel
    ck_assert_int_eq(vine_object_refs((vine_object_s *) phys), 1);

    ck_assert_int_eq(vine_object_refs((vine_object_s *) accel), 1);

    vine_accel_release((vine_accel **) &accel);

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 1);

    ck_assert(handled_tasks(task_handler_state));

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 0);

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PROC), 1);

    vine_proc_put(issue_proc);

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PROC), 0);

    vine_final_exit(vpipe);
} /* START_TEST */

END_TEST START_TEST(test_task_issue_sync)
{
    vine_pipe_s *vpipe = vine_first_init();
    vine_vaccel_s *accel;
    vine_accel_type_e at = _i % VINE_ACCEL_TYPES;

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PROC), 0);

    vine_proc_s *issue_proc = create_proc(vpipe, "issue_proc");

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PROC), 1);

    accel = vine_accel_acquire_type(at);

    ck_assert_int_eq(vine_object_refs((vine_object_s *) accel), 1);

    void *task_handler_state = handle_n_tasks(UTILS_QUEUE_CAPACITY * 2, at);

    for (int cnt = 0; cnt < UTILS_QUEUE_CAPACITY * 2; cnt++) {
        ck_assert_int_eq(vine_task_issue_sync(accel, issue_proc, 0, 0, 0, 0, 0, 0), task_completed);

        ck_assert_int_eq(vine_object_refs((vine_object_s *) accel), 1);
    }

    // Normally scheduler would set phys to something valid.
    ck_assert(accel->phys);

    vine_accel_s *phys = accel->phys;

    // The physical accelerator should only be referenced by accel
    ck_assert_int_eq(vine_object_refs((vine_object_s *) phys), 1);

    ck_assert_int_eq(vine_object_refs((vine_object_s *) accel), 1);

    vine_accel_release((vine_accel **) &accel);

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 1);

    ck_assert(handled_tasks(task_handler_state));

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 0);

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PROC), 1);

    vine_proc_put(issue_proc);

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PROC), 0);

    vine_final_exit(vpipe);
} /* START_TEST */

END_TEST START_TEST(test_type_strings)
{
    switch (_i) {
        case 0 ... VINE_ACCEL_TYPES:
            ck_assert_int_eq(_i, vine_accel_type_from_str(vine_accel_type_to_str(_i)));
            break;
        case VINE_ACCEL_TYPES + 1:
            ck_assert_int_eq(vine_accel_type_from_str("NotRealyAType"), VINE_ACCEL_TYPES);
            ck_assert(!vine_accel_type_to_str(VINE_ACCEL_TYPES));
            ck_assert(!vine_accel_type_to_str(VINE_ACCEL_TYPES + 1));
            break;
    }
}

END_TEST START_TEST(test_empty_task)
{
    vine_pipe_s *vpipe = vine_first_init();

    vine_task *task = vine_task_alloc(vpipe, 0, 0, 0);

    vine_task_mark_done(task, task_completed);

    vine_task_wait(task);

    vine_task_free(task);

    vine_final_exit(vpipe);
}

START_TEST(test_assert_false)
{
    vine_assert(0);
    ck_abort_msg("Should've aborted...");
}
END_TEST START_TEST(test_assert_true)
{
    vine_assert(1);
}

END_TEST

Suite* suite_init()
{
    Suite *s;
    TCase *tc_single;

    s         = suite_create("Vine Talk");
    tc_single = tcase_create("Single");
    tcase_add_unchecked_fixture(tc_single, setup, teardown);
    tcase_add_loop_test(tc_single, test_in_out, 0, 10);
    tcase_add_test(tc_single, test_revision);
    tcase_add_loop_test(tc_single, test_single_proc, 1, VINE_ACCEL_TYPES);
    tcase_add_loop_test(tc_single, test_task_issue_and_wait_v1, 0, VINE_ACCEL_TYPES);
    tcase_add_loop_test(tc_single, test_task_issue_sync, 0, VINE_ACCEL_TYPES);
    tcase_add_loop_test(tc_single, test_type_strings, 0, VINE_ACCEL_TYPES + 2);
    tcase_add_test(tc_single, test_empty_task);
    tcase_add_test_raise_signal(tc_single, test_assert_false, 6);
    tcase_add_test(tc_single, test_assert_true);
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
