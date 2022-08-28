#include "testing.h"
#include "arax_pipe.h"
#include "core/arax_data.h"

TEST_CASE("arax_accel tests")
{
    test_common_setup();

    int fd = test_open_config();

    const char *config = test_create_config(0x1000000);

    write(fd, config, strlen(config) );

    close(fd);

    arax_pipe_s *vpipe = arax_first_init();

    for (arax_accel_type_e type = ANY; type < ARAX_ACCEL_TYPES; type = (arax_accel_type_e) (((int) type) + 1)) {
        DYNAMIC_SECTION("Scenario Tests Type:" << arax_accel_type_to_str(type))
        {
            SECTION("Post Assigned")
            {
                arax_vaccel_s *virt = arax_vaccel_init(vpipe, "VirtAccel", type, 0);

                DYNAMIC_SECTION("get_orphan")
                {
                    REQUIRE(arax_pipe_have_orphan_vaccels(vpipe) == 1);

                    REQUIRE(arax_pipe_get_orphan_vaccel(vpipe) == virt);

                    REQUIRE(arax_pipe_have_orphan_vaccels(vpipe) == 0);

                    arax_accel_release((arax_accel **) &(virt));
                }

                DYNAMIC_SECTION("add_task_at_post_assigned_vac" << arax_accel_type_to_str(type))
                {
                    // Add a fake task, prior to assignment,
                    // to see if it will be 'counted' at the physical accelerator
                    // after assignment
                    arax_vaccel_add_task(virt, virt);

                    arax_accel_s *phys = arax_accel_init(vpipe, "PhysAccel", type, 10, 100);

                    arax_vaccel_s **vacs;

                    // Should have 1 assigned vaccels (free vaq of phys)
                    REQUIRE(arax_accel_get_assigned_vaccels(phys, &vacs) == 1);

                    free(vacs);

                    REQUIRE(arax_accel_pending_tasks(phys) == 0);

                    arax_accel_add_vaccel(phys, virt);

                    // Check to see if task was counted after assignment
                    REQUIRE(arax_accel_pending_tasks(phys) == 1);

                    // Should have 2 assigned vaccels
                    REQUIRE(arax_accel_get_assigned_vaccels(phys, &vacs) == 2);

                    // Should be the one we assigned
                    REQUIRE(vacs[0] == virt);

                    free(vacs);

                    arax_accel_release((arax_accel **) &(virt));

                    // We released the assigned accel, so it should have just the free vaq
                    REQUIRE(arax_accel_get_assigned_vaccels(phys, &vacs) == 1);
                    free(vacs);

                    arax_accel_release((arax_accel **) &(phys));
                } /* START_TEST */
            }

            DYNAMIC_SECTION("assign_at_init" << arax_accel_type_to_str(type))
            {
                arax_accel_s *phys = arax_accel_init(vpipe, "PhysAccel", type, 10, 100);

                // Assign at init time
                arax_vaccel_s *virt = arax_vaccel_init(vpipe, "VirtAccel", type, phys);

                REQUIRE(arax_pipe_have_orphan_vaccels(vpipe) == 0);

                // Should have two assigned(virt + free)
                arax_vaccel_s **vacs;

                REQUIRE(arax_accel_get_assigned_vaccels(phys, &vacs) == 2);
                free(vacs);

                arax_accel_release((arax_accel **) &(virt));

                // We released the assigned accel, so it should jus the free vaq
                REQUIRE(arax_accel_get_assigned_vaccels(phys, &vacs) == 1);
                free(vacs);

                arax_accel_release((arax_accel **) &(phys));
            }

            DYNAMIC_SECTION("add_task_at_pre_assigned_vac" << arax_accel_type_to_str(type))
            {
                arax_accel_s *phys = arax_accel_init(vpipe, "PhysAccel", type, 10, 100);

                // Assign at init time
                arax_vaccel_s *virt = arax_vaccel_init(vpipe, "VirtAccel", type, phys);

                REQUIRE(arax_pipe_have_orphan_vaccels(vpipe) == 0);
                REQUIRE(arax_accel_pending_tasks(phys) == 0);

                arax_vaccel_add_task(virt, virt);

                REQUIRE(arax_accel_pending_tasks(phys) == 1);

                arax_accel_release((arax_accel **) &(virt));
                arax_accel_release((arax_accel **) &(phys));
            }
        }

        DYNAMIC_SECTION("test_single_accel" << arax_accel_type_to_str(type) )
        {
            int accels;
            int cnt;
            arax_accel **accel_ar = 0;
            arax_accel_s *accel;
            arax_accel *vaccel, *vaccel_temp;


            for (cnt = 0; cnt < ARAX_ACCEL_TYPES; cnt++) {
                accels = arax_accel_list((arax_accel_type_e) cnt, 1, 0);
                REQUIRE(accels == 0);
            }

            accel = (arax_accel_s *) arax_accel_acquire_type(type);
            REQUIRE(accel);

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_VIRT_ACCEL) == 1);
            arax_accel_release((arax_accel **) &accel);
            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_VIRT_ACCEL) == 0);

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 0);
            accel = arax_accel_init(vpipe, "FakeAccel", type, 10, 100);
            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 1);

            REQUIRE(accel);
            REQUIRE(arax_accel_get_revision(accel) == 1);
            REQUIRE(arax_object_refs(&(accel->obj)) == 1);

            for (cnt = 0; cnt < ARAX_ACCEL_TYPES; cnt++) {
                REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 1);
                accels = arax_accel_list((arax_accel_type_e) cnt, 1, &accel_ar);
                if (cnt == type || !cnt)
                    REQUIRE(arax_object_refs(&(accel->obj)) == 2);
                else
                    REQUIRE(arax_object_refs(&(accel->obj)) == 1);
                REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 1);
                if (cnt == type || !cnt) {
                    REQUIRE(accels == 1);
                    if (cnt) {
                        REQUIRE(arax_accel_type(
                              accel_ar[0]) == cnt);
                    }
                    REQUIRE(accel == accel_ar[0]);
                    REQUIRE(arax_accel_stat(accel_ar[0], 0) == accel_idle);

                    /* Lets get virtual! */
                    REQUIRE(arax_accel_list(ANY, 0, 0) == 1);
                    vaccel = accel_ar[0];

                    REQUIRE(arax_accel_acquire_phys(&vaccel));
                    REQUIRE(arax_accel_list(ANY, 0, 0) == 2);
                    if (cnt)
                        REQUIRE(arax_accel_list((arax_accel_type_e) cnt, 0, 0) == 1 + (cnt == type));
                    REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_VIRT_ACCEL) == 2);
                    REQUIRE(arax_accel_get_revision(accel) == 2 + (!!cnt) * 2);
                    /* got virtual accel */
                    REQUIRE(((arax_accel_s *) (vaccel))->obj.type ==
                      ARAX_TYPE_VIRT_ACCEL);
                    REQUIRE(arax_vaccel_get_ordering((arax_accel_s *) vaccel) == SEQUENTIAL);
                    arax_vaccel_set_ordering((arax_accel_s *) vaccel, PARALLEL);
                    REQUIRE(arax_vaccel_get_ordering((arax_accel_s *) vaccel) == PARALLEL);
                    arax_vaccel_set_ordering((arax_accel_s *) vaccel, SEQUENTIAL);
                    REQUIRE(arax_vaccel_get_ordering((arax_accel_s *) vaccel) == SEQUENTIAL);
                    REQUIRE(arax_vaccel_queue_size((arax_vaccel_s *) vaccel) != -1);
                    arax_vaccel_set_cid((arax_vaccel_s *) vaccel, 123);
                    REQUIRE(arax_vaccel_get_cid((arax_vaccel_s *) vaccel) == 123);
                    arax_vaccel_set_cid((arax_vaccel_s *) vaccel, 0);
                    REQUIRE(arax_vaccel_get_cid((arax_vaccel_s *) vaccel) == 0);

                    arax_vaccel_set_job_priority((arax_vaccel_s *) vaccel, 123);
                    REQUIRE(arax_vaccel_get_job_priority((arax_vaccel_s *) vaccel) == 123);
                    arax_vaccel_set_job_priority((arax_vaccel_s *) vaccel, 0);
                    REQUIRE(arax_vaccel_get_job_priority((arax_vaccel_s *) vaccel) == 0);

                    arax_vaccel_set_meta((arax_vaccel_s *) vaccel, 0);
                    REQUIRE(arax_vaccel_get_meta((arax_vaccel_s *) vaccel) == 0);
                    arax_vaccel_set_meta((arax_vaccel_s *) vaccel, (void *) 0xF00F);
                    REQUIRE(arax_vaccel_get_meta((arax_vaccel_s *) vaccel) == (void *) 0xF00F);

                    REQUIRE(arax_vaccel_queue(((arax_vaccel_s *) (vaccel))) != 0);
                    REQUIRE(arax_vaccel_queue_size(((arax_vaccel_s *) (vaccel))) == 0);
                    /* Cant get a virtual out of a virtual accel */
                    REQUIRE(!arax_accel_acquire_phys(&vaccel));
                    REQUIRE(arax_accel_stat(vaccel, 0) == accel_idle);

                    vaccel_temp = vaccel;
                    REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_VIRT_ACCEL) == 2);
                    arax_accel_release(&(vaccel_temp));
                    REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_VIRT_ACCEL) == 1);
                    REQUIRE(arax_accel_get_revision(accel) == 3 + (!!cnt) * 2);
                } else {
                    REQUIRE(accels == 0);
                }
                if (cnt == type || !cnt) {
                    REQUIRE(arax_object_refs(&(accel->obj)) == 2);
                }
            }
            arax_accel_list_free(accel_ar);
            REQUIRE(arax_object_refs(&(accel->obj)) == 1);
            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 1);
            REQUIRE(!arax_pipe_delete_accel(vpipe, accel) );
            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 0);
            REQUIRE(arax_pipe_delete_accel(vpipe, accel) );
            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 0);
        } /* START_TEST */
    }

    arax_final_exit(vpipe);

    test_common_teardown();
}
