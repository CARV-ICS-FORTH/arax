#include "testing.h"
#include "vine_pipe.h"
#include "core/vine_data.h"

TEST_CASE("vine_accel tests")
{
    test_common_setup();

    int fd = test_open_config();

    const char *config = test_create_config(0x1000000);

    write(fd, config, strlen(config) );

    close(fd);

    vine_pipe_s *vpipe = vine_first_init();

    for (vine_accel_type_e type = ANY; type < VINE_ACCEL_TYPES; type = (vine_accel_type_e) (((int) type) + 1)) {
        DYNAMIC_SECTION("Scenario Tests Type:" << vine_accel_type_to_str(type))
        {
            SECTION("Post Assigned")
            {
                vine_vaccel_s *virt = vine_vaccel_init(vpipe, "VirtAccel", type, 0);

                DYNAMIC_SECTION("get_orphan")
                {
                    REQUIRE(vine_pipe_have_orphan_vaccels(vpipe) == 1);

                    REQUIRE(vine_pipe_get_orphan_vaccel(vpipe) == virt);

                    REQUIRE(vine_pipe_have_orphan_vaccels(vpipe) == 0);

                    vine_accel_release((vine_accel **) &(virt));
                }

                DYNAMIC_SECTION("add_task_at_post_assigned_vac" << vine_accel_type_to_str(type))
                {
                    // Add a fake task, prior to assignment,
                    // to see if it will be 'counted' at the physical accelerator
                    // after assignment
                    vine_vaccel_add_task(virt, virt);

                    vine_accel_s *phys = vine_accel_init(vpipe, "PhysAccel", type, 10, 100);

                    vine_vaccel_s **vacs;

                    // Should have no assigned vaccels
                    REQUIRE(vine_accel_get_assigned_vaccels(phys, &vacs) == 0);

                    free(vacs);

                    REQUIRE(vine_accel_pending_tasks(phys) == 0);

                    vine_accel_add_vaccel(phys, virt);

                    // Check to see if task was counted after assignment
                    REQUIRE(vine_accel_pending_tasks(phys) == 1);

                    // Should have 1 assigned vaccels
                    REQUIRE(vine_accel_get_assigned_vaccels(phys, &vacs) == 1);

                    // Should be the one we assigned
                    REQUIRE(vacs[0] == virt);

                    free(vacs);

                    vine_accel_release((vine_accel **) &(virt));

                    // We released the assigned accel, so it should have no assigned accels
                    REQUIRE(vine_accel_get_assigned_vaccels(phys, &vacs) == 0);

                    vine_accel_release((vine_accel **) &(phys));
                } /* START_TEST */
            }

            DYNAMIC_SECTION("assign_at_init" << vine_accel_type_to_str(type))
            {
                vine_accel_s *phys = vine_accel_init(vpipe, "PhysAccel", type, 10, 100);

                // Assign at init time
                vine_vaccel_s *virt = vine_vaccel_init(vpipe, "VirtAccel", type, phys);

                REQUIRE(vine_pipe_have_orphan_vaccels(vpipe) == 0);

                // Should have one assigned
                vine_vaccel_s **vacs;

                REQUIRE(vine_accel_get_assigned_vaccels(phys, &vacs) == 1);
                free(vacs);

                vine_accel_release((vine_accel **) &(virt));

                // We released the assigned accel, so it should have no assigned accels
                REQUIRE(vine_accel_get_assigned_vaccels(phys, &vacs) == 0);
                free(vacs);

                vine_accel_release((vine_accel **) &(phys));
            }

            DYNAMIC_SECTION("add_task_at_pre_assigned_vac" << vine_accel_type_to_str(type))
            {
                vine_accel_s *phys = vine_accel_init(vpipe, "PhysAccel", type, 10, 100);

                // Assign at init time
                vine_vaccel_s *virt = vine_vaccel_init(vpipe, "VirtAccel", type, phys);

                REQUIRE(vine_pipe_have_orphan_vaccels(vpipe) == 0);
                REQUIRE(vine_accel_pending_tasks(phys) == 0);

                vine_vaccel_add_task(virt, virt);

                REQUIRE(vine_accel_pending_tasks(phys) == 1);

                vine_accel_release((vine_accel **) &(virt));
                vine_accel_release((vine_accel **) &(phys));
            }
        }

        DYNAMIC_SECTION("test_single_accel" << vine_accel_type_to_str(type) )
        {
            int accels;
            int cnt;
            vine_accel **accel_ar = 0;
            vine_accel_s *accel;
            vine_accel *vaccel, *vaccel_temp;


            for (cnt = 0; cnt < VINE_ACCEL_TYPES; cnt++) {
                accels = vine_accel_list((vine_accel_type_e) cnt, 1, 0);
                REQUIRE(accels == 0);
            }

            accel = (vine_accel_s *) vine_accel_acquire_type(type);
            REQUIRE(accel);

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL) == 1);
            vine_accel_release((vine_accel **) &accel);
            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL) == 0);

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL) == 0);
            accel = vine_accel_init(vpipe, "FakeAccel", type, 10, 100);
            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL) == 1);

            REQUIRE(accel);
            REQUIRE(vine_accel_get_revision(accel) == 0);
            REQUIRE(vine_object_refs(&(accel->obj)) == 1);

            for (cnt = 0; cnt < VINE_ACCEL_TYPES; cnt++) {
                REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL) == 1);
                accels = vine_accel_list((vine_accel_type_e) cnt, 1, &accel_ar);
                if (cnt == type || !cnt)
                    REQUIRE(vine_object_refs(&(accel->obj)) == 2);
                else
                    REQUIRE(vine_object_refs(&(accel->obj)) == 1);
                REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL) == 1);
                if (cnt == type || !cnt) {
                    REQUIRE(accels == 1);
                    if (cnt) {
                        REQUIRE(vine_accel_type(
                              accel_ar[0]) == cnt);
                    }
                    REQUIRE(accel == accel_ar[0]);
                    REQUIRE(vine_accel_stat(accel_ar[0], 0) == accel_idle);

                    /* Lets get virtual! */
                    REQUIRE(vine_accel_list(ANY, 0, 0) == 0);
                    vaccel = accel_ar[0];

                    REQUIRE(vine_accel_acquire_phys(&vaccel));
                    REQUIRE(vine_accel_list(ANY, 0, 0) == 1);
                    if (cnt)
                        REQUIRE(vine_accel_list((vine_accel_type_e) cnt, 0, 0) == (cnt == type));
                    REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL) == 1);
                    REQUIRE(vine_accel_get_revision(accel) == 1 + (!!cnt) * 2);
                    /* got virtual accel */
                    REQUIRE(((vine_accel_s *) (vaccel))->obj.type ==
                      VINE_TYPE_VIRT_ACCEL);
                    REQUIRE(vine_vaccel_get_ordering((vine_accel_s *) vaccel) == SEQUENTIAL);
                    vine_vaccel_set_ordering((vine_accel_s *) vaccel, PARALLEL);
                    REQUIRE(vine_vaccel_get_ordering((vine_accel_s *) vaccel) == PARALLEL);
                    vine_vaccel_set_ordering((vine_accel_s *) vaccel, SEQUENTIAL);
                    REQUIRE(vine_vaccel_get_ordering((vine_accel_s *) vaccel) == SEQUENTIAL);
                    REQUIRE(vine_vaccel_queue_size((vine_vaccel_s *) vaccel) != -1);
                    vine_vaccel_set_cid((vine_vaccel_s *) vaccel, 123);
                    REQUIRE(vine_vaccel_get_cid((vine_vaccel_s *) vaccel) == 123);
                    vine_vaccel_set_cid((vine_vaccel_s *) vaccel, 0);
                    REQUIRE(vine_vaccel_get_cid((vine_vaccel_s *) vaccel) == 0);

                    vine_vaccel_set_job_priority((vine_vaccel_s *) vaccel, 123);
                    REQUIRE(vine_vaccel_get_job_priority((vine_vaccel_s *) vaccel) == 123);
                    vine_vaccel_set_job_priority((vine_vaccel_s *) vaccel, 0);
                    REQUIRE(vine_vaccel_get_job_priority((vine_vaccel_s *) vaccel) == 0);

                    vine_vaccel_set_meta((vine_vaccel_s *) vaccel, 0);
                    REQUIRE(vine_vaccel_get_meta((vine_vaccel_s *) vaccel) == 0);
                    vine_vaccel_set_meta((vine_vaccel_s *) vaccel, (void *) 0xF00F);
                    REQUIRE(vine_vaccel_get_meta((vine_vaccel_s *) vaccel) == (void *) 0xF00F);

                    REQUIRE(vine_vaccel_queue(((vine_vaccel_s *) (vaccel))) != 0);
                    REQUIRE(vine_vaccel_queue_size(((vine_vaccel_s *) (vaccel))) == 0);
                    /* Cant get a virtual out of a virtual accel */
                    REQUIRE(!vine_accel_acquire_phys(&vaccel));
                    REQUIRE(vine_accel_stat(vaccel, 0) == accel_idle);

                    vaccel_temp = vaccel;
                    REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL) == 1);
                    vine_accel_release(&(vaccel_temp));
                    REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL) == 0);
                    REQUIRE(vine_accel_get_revision(accel) == 2 + (!!cnt) * 2);
                } else {
                    REQUIRE(accels == 0);
                }
                if (cnt == type || !cnt) {
                    REQUIRE(vine_object_refs(&(accel->obj)) == 2);
                }
            }
            vine_accel_list_free(accel_ar);
            REQUIRE(vine_object_refs(&(accel->obj)) == 1);
            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL) == 1);
            REQUIRE(!vine_pipe_delete_accel(vpipe, accel) );
            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL) == 0);
            REQUIRE(vine_pipe_delete_accel(vpipe, accel) );
            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL) == 0);
        } /* START_TEST */
    }

    vine_final_exit(vpipe);

    test_common_teardown();
}
