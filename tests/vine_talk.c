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

END_TEST START_TEST(test_single_accel)
{
    int accels;
    int cnt;
    vine_accel **accel_ar = 0;
    vine_accel_s *accel;
    vine_accel *vaccel, *vaccel_temp;
    vine_pipe_s *vpipe = vine_first_init();

    for (cnt = 0; cnt < VINE_ACCEL_TYPES; cnt++) {
        accels = vine_accel_list(cnt, 1, 0);
        ck_assert_int_eq(accels, 0);
    }

    accel = vine_accel_acquire_type(_i);
    ck_assert(!!accel);

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL), 1);
    vine_accel_release((vine_accel **) &accel);
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL), 0);

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 0);
    accel = vine_accel_init(vpipe, "FakeAccel", _i, 10, 100);
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 1);

    ck_assert(!!accel);
    ck_assert_int_eq(vine_accel_get_revision(accel), 0);
    ck_assert_int_eq(vine_object_refs(&(accel->obj)), 1);

    vine_accel_location(accel);

    for (cnt = 0; cnt < VINE_ACCEL_TYPES; cnt++) {
        ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 1);
        accels = vine_accel_list(cnt, 1, &accel_ar);
        if (cnt == _i || !cnt)
            ck_assert_int_eq(vine_object_refs(&(accel->obj)), 2);
        else
            ck_assert_int_eq(vine_object_refs(&(accel->obj)), 1);
        ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 1);
        if (cnt == _i || !cnt) {
            ck_assert_int_eq(accels, 1);
            if (cnt) {
                ck_assert_int_eq(vine_accel_type(
                      accel_ar[0]), cnt);
            }
            ck_assert_ptr_eq(accel, accel_ar[0]);
            ck_assert_int_eq(vine_accel_stat(accel_ar[0], 0), accel_idle);

            /* Lets get virtual! */
            ck_assert_int_eq(vine_accel_list(ANY, 0, 0), 0);
            vaccel = accel_ar[0];

            ck_assert(vine_accel_acquire_phys(&vaccel));
            ck_assert_int_eq(vine_accel_list(ANY, 0, 0), 1);
            ck_assert_int_eq(vine_accel_list(cnt, 0, 0), (cnt == _i) || (cnt == 0));
            ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL), 1);
            ck_assert_int_eq(vine_accel_get_revision(accel), 1 + (!!cnt) * 2);
            /* got virtual accel */
            ck_assert_int_eq(((vine_accel_s *) (vaccel))->obj.type,
              VINE_TYPE_VIRT_ACCEL);
            ck_assert_int_eq(vine_vaccel_get_ordering(vaccel), SEQUENTIAL);
            vine_vaccel_set_ordering(vaccel, PARALLEL);
            ck_assert_int_eq(vine_vaccel_get_ordering(vaccel), PARALLEL);
            vine_vaccel_set_ordering(vaccel, SEQUENTIAL);
            ck_assert_int_eq(vine_vaccel_get_ordering(vaccel), SEQUENTIAL);
            ck_assert(vine_vaccel_queue_size(vaccel) != -1);
            vine_vaccel_set_cid(vaccel, 123);
            ck_assert_int_eq(vine_vaccel_get_cid(vaccel), 123);
            vine_vaccel_set_cid(vaccel, 0);
            ck_assert_int_eq(vine_vaccel_get_cid(vaccel), 0);

            vine_vaccel_set_job_priority(vaccel, 123);
            ck_assert_int_eq(vine_vaccel_get_job_priority(vaccel), 123);
            vine_vaccel_set_job_priority(vaccel, 0);
            ck_assert_int_eq(vine_vaccel_get_job_priority(vaccel), 0);

            ck_assert_ptr_eq(vine_vaccel_get_assignee(vaccel), 0);                       // Initially not assigned
            ck_assert_ptr_eq(vine_vaccel_test_set_assignee(vaccel, accel), accel);       // First set should work
            ck_assert_ptr_eq(vine_vaccel_test_set_assignee(vaccel, accel), accel);       // Same set should work
            ck_assert_ptr_eq(vine_vaccel_test_set_assignee(vaccel, (void *) 0xBAAD), 0); // Different set should fail

            vine_vaccel_set_meta(vaccel, 0);
            ck_assert_ptr_eq(vine_vaccel_get_meta(vaccel), 0);
            vine_vaccel_set_meta(vaccel, (void *) 0xF00F);
            ck_assert_ptr_eq(vine_vaccel_get_meta(vaccel), (void *) 0xF00F);

            vine_vaccel_mark_task_done(vaccel);
            vine_vaccel_wait_task_done(vaccel);

            ck_assert(vine_vaccel_queue(((vine_vaccel_s *) (vaccel))) != 0);
            ck_assert(vine_vaccel_queue_size(((vine_vaccel_s *) (vaccel))) == 0);
            /* Cant get a virtual out of a virtual accel */
            ck_assert(!vine_accel_acquire_phys(&vaccel));
            ck_assert_int_eq(vine_accel_stat(vaccel, 0), accel_idle);
            vine_accel_location(vaccel);

            vaccel_temp = vaccel;
            ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL), 1);
            vine_accel_release(&(vaccel_temp));
            ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL), 0);
            ck_assert_int_eq(vine_accel_get_revision(accel), 2 + (!!cnt) * 2);
        } else {
            ck_assert_int_eq(accels, 0);
        }
        if (cnt == _i || !cnt) {
            ck_assert_int_eq(vine_object_refs(&(accel->obj)), 2);
        }
    }
    vine_accel_list_free(accel_ar);
    ck_assert_int_eq(vine_object_refs(&(accel->obj)), 1);
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 1);
    ck_assert(!vine_pipe_delete_accel(vpipe, accel) );
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 0);
    ck_assert(vine_pipe_delete_accel(vpipe, accel) );
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 0);

    vine_final_exit(vpipe);
    /* setup()/teardown() */
} /* START_TEST */

END_TEST

vine_proc_s* create_proc(vine_pipe_s *vpipe, int type, const char *name, void *pd, size_t psize)
{
    vine_proc_s *proc;

    ck_assert(!!vpipe);
    ck_assert(!vine_proc_get(type, name) );
    proc = (vine_proc_s *) vine_proc_register(type, name, pd, psize);
    return proc;
}

START_TEST(test_single_proc)
{
    int cnt;
    size_t cs;
    vine_proc_s *proc;
    vine_pipe_s *vpipe = vine_first_init();
    char pd[] = "TEST_DATA";

    proc = create_proc(vpipe, _i, "TEST_PROC", pd, _i);

    if (_i) {
        ck_assert(!!proc);
    } else { /* Fail to create an ANY procedure */
        ck_assert(!proc);
        ck_assert(!vine_proc_get(_i, "TEST_PROC") );
        vine_final_exit(vpipe);
        return;
    }

    ck_assert_ptr_eq(vine_proc_register(_i, "TEST_PROC", "DIFF_DATA", _i), 0);

    ck_assert(vine_proc_get_code(proc, &cs) != NULL);
    ck_assert_int_eq(cs, _i);
    ck_assert(vine_proc_match_code(proc, pd, _i) );
    ck_assert(!vine_proc_match_code(proc, pd, _i - 1) );

    VineFunctor *vf = vine_proc_get_functor(proc);

    ck_assert(strncmp((char *) &vf, pd, _i) == 0);

    for (cnt = 0; cnt < VINE_ACCEL_TYPES; cnt++) {
        if (cnt == _i || cnt == ANY) {
            ck_assert(vine_proc_get(cnt, "TEST_PROC") != NULL);
            ck_assert(vine_proc_put(proc) );
        } else {
            ck_assert(!vine_proc_get(cnt, "TEST_PROC") );
        }
    }

    vine_proc_put(proc);

    ck_assert(!vine_proc_get(_i, "TEST_PROC") );

    vine_final_exit(vpipe);
}
END_TEST START_TEST(test_task_issue_and_wait_v1)
{
    vine_pipe_s *vpipe = vine_first_init();
    vine_vaccel_s *accel;
    vine_accel_type_e at = _i % VINE_ACCEL_TYPES;

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PROC), 0);

    vine_proc_s *issue_proc = create_proc(vpipe, at, "issue_proc", 0, 0);

    if (at == ANY) {
        ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PROC), 0);
        ck_assert(!issue_proc);

        ck_assert(!vine_proc_get(at, "TEST_PROC") );
        vine_final_exit(vpipe);
        return;
    }

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PROC), 1);

    accel = vine_accel_acquire_type(at);

    ck_assert_int_eq(vine_object_refs((vine_object_s *) accel), 1);

    // I expect 2 tasks (init_phys for args and issue_proc)
    void *task_handler_state = handle_n_tasks(1, at);

    vine_task *task = vine_task_issue(accel, issue_proc, 0, 0, 0, 0, 0, 0);

    ck_assert_int_eq(vine_object_refs((vine_object_s *) accel), 2);

    vine_task_wait_done(task);

    ck_assert_int_eq(vine_task_stat(task, 0), task_completed);

    vine_task_free(task);

    ck_assert_int_eq(vine_object_refs((vine_object_s *) accel), 1);

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

    vine_proc_s *issue_proc = create_proc(vpipe, at, "issue_proc", 0, 0);

    if (at == ANY) {
        ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PROC), 0);
        ck_assert(!issue_proc);

        ck_assert(!vine_proc_get(at, "TEST_PROC") );
        vine_final_exit(vpipe);
        return;
    }

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PROC), 1);

    accel = vine_accel_acquire_type(at);

    ck_assert_int_eq(vine_object_refs((vine_object_s *) accel), 1);

    // I expect 2 tasks (init_phys for args and issue_proc)
    void *task_handler_state = handle_n_tasks(1, at);

    ck_assert_int_eq(vine_task_issue_sync(accel, issue_proc, 0, 0, 0, 0, 0, 0), task_completed);

    ck_assert_int_eq(vine_object_refs((vine_object_s *) accel), 1);

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

END_TEST START_TEST(test_vac_ordering)
{
    vine_pipe_s *vpipe = vine_first_init();

    vine_accel_s *vaccel = (vine_accel_s *) vine_vaccel_init(vpipe, "Test", GPU, 0);

    ck_assert_int_eq(vine_vaccel_get_ordering(vaccel), SEQUENTIAL);

    // Test original behaviour

    ck_assert_ptr_eq(vine_vaccel_get_assignee(vaccel), 0);

    ck_assert_ptr_eq(vine_vaccel_test_set_assignee(vaccel, 0), 0);

    // First set should be succesfull
    ck_assert_ptr_eq(vine_vaccel_test_set_assignee(vaccel, (void *) 0xF00F), (void *) 0xF00F);

    // Second set to different address should be fail
    ck_assert_ptr_eq(vine_vaccel_test_set_assignee(vaccel, (void *) 0xFFFF), 0x0);

    // Same set should be succesfull
    ck_assert_ptr_eq(vine_vaccel_test_set_assignee(vaccel, (void *) 0xF00F), (void *) 0xF00F);

    // Reset assignee
    ((vine_vaccel_s *) vaccel)->assignee = 0;

    vine_vaccel_set_ordering(vaccel, PARALLEL);
    ck_assert_int_eq(vine_vaccel_get_ordering(vaccel), PARALLEL);

    ck_assert_ptr_eq(vine_vaccel_get_assignee(vaccel), 0);

    ck_assert_ptr_eq(vine_vaccel_test_set_assignee(vaccel, 0), 0);

    // Parallel should always work
    ck_assert_ptr_eq(vine_vaccel_test_set_assignee(vaccel, (void *) 0xF00F), (void *) 0xF00F);

    // Parallel should always work
    ck_assert_ptr_eq(vine_vaccel_test_set_assignee(vaccel, (void *) 0xF11F), (void *) 0xF11F);

    vine_accel_release((vine_accel *) &vaccel);

    vine_final_exit(vpipe);
} /* START_TEST */

END_TEST

/*
 * START_TEST(test_assert_false)
 * {
 *  vine_assert(0);
 *  ck_abort_msg("Should've aborted...");
 * }
 * END_TEST
 */
START_TEST(test_assert_true)
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
    tcase_add_loop_test(tc_single, test_single_accel, 0, VINE_ACCEL_TYPES);
    tcase_add_loop_test(tc_single, test_single_proc, 0, VINE_ACCEL_TYPES);
    tcase_add_loop_test(tc_single, test_task_issue_and_wait_v1, 0, VINE_ACCEL_TYPES);
    tcase_add_loop_test(tc_single, test_task_issue_sync, 0, VINE_ACCEL_TYPES);
    tcase_add_loop_test(tc_single, test_type_strings, 0, VINE_ACCEL_TYPES + 2);
    tcase_add_test(tc_single, test_empty_task);
    tcase_add_test(tc_single, test_vac_ordering);
    // tcase_add_test_raise_signal(tc_single, test_assert_false,6);
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
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_NORMAL);
    failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
