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

START_TEST(test_single_accel)
{
    int accels;
    int cnt;
    int type = _i;
    vine_accel **accel_ar = 0;
    vine_accel_s *accel;
    vine_accel *vaccel, *vaccel_temp;
    vine_pipe_s *vpipe = vine_first_init();

    for (cnt = 0; cnt < VINE_ACCEL_TYPES; cnt++) {
        accels = vine_accel_list(cnt, 1, 0);
        ck_assert_int_eq(accels, 0);
    }

    accel = vine_accel_acquire_type(type);
    ck_assert(!!accel);

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL), 1);
    vine_accel_release((vine_accel **) &accel);
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL), 0);

    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 0);
    accel = vine_accel_init(vpipe, "FakeAccel", type, 10, 100);
    ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 1);

    ck_assert(!!accel);
    ck_assert_int_eq(vine_accel_get_revision(accel), 0);
    ck_assert_int_eq(vine_object_refs(&(accel->obj)), 1);

    vine_accel_location(accel);

    for (cnt = 0; cnt < VINE_ACCEL_TYPES; cnt++) {
        ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 1);
        accels = vine_accel_list(cnt, 1, &accel_ar);
        if (cnt == type || !cnt)
            ck_assert_int_eq(vine_object_refs(&(accel->obj)), 2);
        else
            ck_assert_int_eq(vine_object_refs(&(accel->obj)), 1);
        ck_assert_int_eq(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL), 1);
        if (cnt == type || !cnt) {
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
            ck_assert_int_eq(vine_accel_list(cnt, 0, 0), (cnt == type) || (cnt == 0));
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

            vine_vaccel_set_meta(vaccel, 0);
            ck_assert_ptr_eq(vine_vaccel_get_meta(vaccel), 0);
            vine_vaccel_set_meta(vaccel, (void *) 0xF00F);
            ck_assert_ptr_eq(vine_vaccel_get_meta(vaccel), (void *) 0xF00F);

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
        if (cnt == type || !cnt) {
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

END_TEST START_TEST(get_orphan)
{
    vine_pipe_s *vpipe  = vine_first_init();
    vine_vaccel_s *virt = vine_vaccel_init(vpipe, "VirtAccel", _i, 0);

    ck_assert_int_eq(vine_pipe_have_orphan_vaccels(vpipe), 1);

    ck_assert_ptr_eq(vine_pipe_get_orphan_vaccel(vpipe), virt);

    ck_assert_int_eq(vine_pipe_have_orphan_vaccels(vpipe), 0);

    vine_accel_release((vine_accel **) &(virt));

    vine_final_exit(vpipe);
}

END_TEST START_TEST(add_task_at_post_assigned_vac)
{
    vine_pipe_s *vpipe  = vine_first_init();
    vine_vaccel_s *virt = vine_vaccel_init(vpipe, "VirtAccel", _i, 0);

    ck_assert_int_eq(vine_pipe_have_orphan_vaccels(vpipe), 1);

    ck_assert_ptr_eq(vine_pipe_get_orphan_vaccel(vpipe), virt);

    ck_assert_int_eq(vine_pipe_have_orphan_vaccels(vpipe), 0);

    // Add a fake task, prior to assignment,
    // to see if it will be 'counted' at the physical accelerator
    // after assignment
    vine_vaccel_add_task(virt, virt);

    vine_accel_s *phys = vine_accel_init(vpipe, "PhysAccel", _i, 10, 100);

    vine_vaccel_s **vacs;

    // Should have no assigned vaccels
    ck_assert_int_eq(vine_accel_get_assigned_vaccels(phys, &vacs), 0);

    free(vacs);

    ck_assert_int_eq(vine_accel_pending_tasks(phys), 0);

    vine_accel_add_vaccel(phys, virt);

    // Check to see if task was counted after assignment
    ck_assert_int_eq(vine_accel_pending_tasks(phys), 1);

    // Should have 1 assigned vaccels
    ck_assert_int_eq(vine_accel_get_assigned_vaccels(phys, &vacs), 1);

    // Should be the one we assigned
    ck_assert_ptr_eq(vacs[0], virt);

    free(vacs);

    vine_accel_release((vine_accel **) &(virt));

    // We released the assigned accel, so it should have no assigned accels
    ck_assert_int_eq(vine_accel_get_assigned_vaccels(phys, &vacs), 0);

    vine_accel_release((vine_accel **) &(phys));

    vine_final_exit(vpipe);
} /* START_TEST */

END_TEST START_TEST(assign_at_init)
{
    vine_pipe_s *vpipe = vine_first_init();
    vine_accel_s *phys = vine_accel_init(vpipe, "PhysAccel", _i, 10, 100);

    // Assign at init time
    vine_vaccel_s *virt = vine_vaccel_init(vpipe, "VirtAccel", _i, phys);

    ck_assert_int_eq(vine_pipe_have_orphan_vaccels(vpipe), 0);

    // Should have one assigned
    vine_vaccel_s **vacs;

    ck_assert_int_eq(vine_accel_get_assigned_vaccels(phys, &vacs), 1);
    free(vacs);

    vine_accel_release((vine_accel **) &(virt));

    // We released the assigned accel, so it should have no assigned accels
    ck_assert_int_eq(vine_accel_get_assigned_vaccels(phys, &vacs), 0);
    free(vacs);

    vine_accel_release((vine_accel **) &(phys));

    vine_final_exit(vpipe);
}

END_TEST START_TEST(add_task_at_pre_assigned_vac)
{
    vine_pipe_s *vpipe = vine_first_init();
    vine_accel_s *phys = vine_accel_init(vpipe, "PhysAccel", _i, 10, 100);

    // Assign at init time
    vine_vaccel_s *virt = vine_vaccel_init(vpipe, "VirtAccel", _i, phys);

    ck_assert_int_eq(vine_pipe_have_orphan_vaccels(vpipe), 0);
    ck_assert_int_eq(vine_accel_pending_tasks(phys), 0);

    vine_vaccel_add_task(virt, virt);

    ck_assert_int_eq(vine_accel_pending_tasks(phys), 1);

    vine_accel_release((vine_accel **) &(virt));
    vine_accel_release((vine_accel **) &(phys));

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
    tcase_add_loop_test(tc_single, test_single_accel, 0, VINE_ACCEL_TYPES);
    tcase_add_loop_test(tc_single, get_orphan, 0, VINE_ACCEL_TYPES);
    tcase_add_loop_test(tc_single, assign_at_init, 0, VINE_ACCEL_TYPES);
    tcase_add_loop_test(tc_single, add_task_at_pre_assigned_vac, 0, VINE_ACCEL_TYPES);
    tcase_add_loop_test(tc_single, add_task_at_post_assigned_vac, 0, VINE_ACCEL_TYPES);
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
