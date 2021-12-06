#include "testing.h"
#include "vine_pipe.h"
#include "core/vine_data.h"

void setup()
{ }

void teardown()
{ }

TEST_CASE("vine_talk tests")
{
    test_common_setup();

    int fd = test_open_config();

    const char *config = test_create_config(0x1000000);

    write(fd, config, strlen(config) );

    close(fd);

    vine_pipe_s *vpipe = vine_first_init();

    vine_talk_controller_init_done();

    SECTION("test_in_out"){
        vine_pipe_s *vpipe2 = vine_talk_init();

        REQUIRE(!!vpipe2);
        REQUIRE(vpipe == vpipe2);

        REQUIRE(vine_pipe_mmap_address(vpipe) == vpipe);
        REQUIRE(vine_pipe_have_to_mmap(vpipe, system_process_id()) == 0);
        vine_talk_exit();
        REQUIRE(vine_pipe_mmap_address(vpipe) == vpipe);
        REQUIRE(vine_pipe_have_to_mmap(vpipe, system_process_id()) == 0);
    }

    SECTION("test_revision")
    {
        char *rev;

        /**
         * Break sha intentionally
         */
        rev = (char *) vine_pipe_get_revision(vpipe);
        REQUIRE(rev);
        REQUIRE(std::string(rev) == VINE_TALK_GIT_REV);
    }

    for (int etype = 0; etype < VINE_ACCEL_TYPES; etype++) {
        vine_accel_type_e vtype = (vine_accel_type_e) etype;
        if (etype) {
            DYNAMIC_SECTION("test_single_proc" << vine_accel_type_to_str(vtype))
            {
                vine_proc_s *proc;
                char pd[] = "TEST_FUNCTOR";
                int cnt;

                proc = create_proc(vpipe, "TEST_PROC");

                REQUIRE(vine_proc_register("TEST_PROC") == proc);

                REQUIRE(vine_proc_can_run_at(proc, vtype) == 0);

                vine_proc_set_functor(proc, vtype, (VineFunctor *) pd);

                REQUIRE(vine_proc_can_run_at(proc, vtype) == 1);

                VineFunctor *vf = vine_proc_get_functor(proc, vtype);

                REQUIRE(vf == (VineFunctor *) pd);

                for (cnt = !ANY; cnt < VINE_ACCEL_TYPES; cnt++) {
                    if (cnt == vtype) {
                        REQUIRE(vine_proc_get_functor(proc, (vine_accel_type_e) cnt) != NULL);
                    } else {
                        REQUIRE(!vine_proc_get_functor(proc, (vine_accel_type_e) cnt) );
                    }
                    REQUIRE(vine_proc_can_run_at(proc, (vine_accel_type_e) cnt) == (cnt == vtype) );
                }

                REQUIRE(!vine_proc_put(proc) );

                REQUIRE(!vine_proc_get("TEST_PROC") );
            } /* START_TEST */
        }

        DYNAMIC_SECTION("test_task_issue_and_wait_v1" << vine_accel_type_to_str(vtype))
        {
            vine_vaccel_s *accel;

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PROC) == 0);

            vine_proc_s *issue_proc = create_proc(vpipe, "issue_proc");

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PROC) == 1);

            accel = (vine_vaccel_s *) vine_accel_acquire_type(vtype);

            REQUIRE(vine_object_refs((vine_object_s *) accel) == 1);

            void *task_handler_state = handle_n_tasks(UTILS_QUEUE_CAPACITY * 2, vtype);

            for (int cnt = 0; cnt < UTILS_QUEUE_CAPACITY * 2; cnt++) {
                vine_task *task = vine_task_issue(accel, issue_proc, 0, 0, 0, 0, 0, 0);

                REQUIRE(vine_object_refs((vine_object_s *) accel) == 2);

                vine_task_wait_done((vine_task_msg_s *) task);

                REQUIRE(vine_task_stat(task, 0) == task_completed);

                vine_task_free(task);

                REQUIRE(vine_object_refs((vine_object_s *) accel) == 1);
            }

            // Normally scheduler would set phys to something valid.
            REQUIRE(accel->phys);

            vine_accel_s *phys = accel->phys;

            // The physical accelerator should only be referenced by accel
            REQUIRE(vine_object_refs((vine_object_s *) phys) == 1);

            REQUIRE(vine_object_refs((vine_object_s *) accel) == 1);

            vine_accel_release((vine_accel **) &accel);

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL) == 1);

            REQUIRE(handled_tasks(task_handler_state));

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL) == 0);

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PROC) == 1);

            vine_proc_put(issue_proc);

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PROC) == 0);
        } /* START_TEST */

        DYNAMIC_SECTION("test_task_issue_sync" << vine_accel_type_to_str(vtype))
        {
            vine_vaccel_s *accel;

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PROC) == 0);

            vine_proc_s *issue_proc = create_proc(vpipe, "issue_proc");

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PROC) == 1);

            accel = (vine_vaccel_s *) vine_accel_acquire_type(vtype);

            REQUIRE(vine_object_refs((vine_object_s *) accel) == 1);

            void *task_handler_state = handle_n_tasks(UTILS_QUEUE_CAPACITY * 2, vtype);

            for (int cnt = 0; cnt < UTILS_QUEUE_CAPACITY * 2; cnt++) {
                REQUIRE(vine_task_issue_sync(accel, issue_proc, 0, 0, 0, 0, 0, 0) == task_completed);

                REQUIRE(vine_object_refs((vine_object_s *) accel) == 1);
            }

            // Normally scheduler would set phys to something valid.
            REQUIRE(accel->phys);

            vine_accel_s *phys = accel->phys;

            // The physical accelerator should only be referenced by accel
            REQUIRE(vine_object_refs((vine_object_s *) phys) == 1);

            REQUIRE(vine_object_refs((vine_object_s *) accel) == 1);

            vine_accel_release((vine_accel **) &accel);

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL) == 1);

            REQUIRE(handled_tasks(task_handler_state));

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL) == 0);

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PROC) == 1);

            vine_proc_put(issue_proc);

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PROC) == 0);
        } /* START_TEST */
    }
    for (vine_accel_type_e type = ANY; type < VINE_ACCEL_TYPES + 2; type = (vine_accel_type_e) (((int) type) + 1)) {
        DYNAMIC_SECTION("test_type_strings" << vine_accel_type_to_str(type))
        {
            switch (type) {
                case 0 ... VINE_ACCEL_TYPES:
                    REQUIRE(type == vine_accel_type_from_str(vine_accel_type_to_str(type)));
                    break;
                case VINE_ACCEL_TYPES + 1:
                    REQUIRE(vine_accel_type_from_str("NotRealyAType") == VINE_ACCEL_TYPES);
                    REQUIRE(!vine_accel_type_to_str(VINE_ACCEL_TYPES));
                    REQUIRE(!vine_accel_type_to_str((vine_accel_type_e) (VINE_ACCEL_TYPES + 1)));
                    break;
            }
        }
    }

    SECTION("test_empty_task")
    {
        vine_pipe_s *vpipe = vine_first_init();

        vine_proc_s *proc = create_proc(vpipe, "test_proc");

        vine_vaccel_s *vac = (vine_vaccel_s *) vine_accel_acquire_type(ANY);

        vine_task *task = vine_task_alloc(vpipe, vac, proc, 0, 0, 0, 0, 0);

        vine_task_mark_done((vine_task_msg_s *) task, task_completed);

        vine_task_wait(task);

        vine_task_free(task);

        vine_object_ref_dec(&(proc->obj));

        vine_final_exit(vpipe);
    }

    /**
     * This test can not be potred to Catch2 as it has to abort(signal 6).
     * Catch2 at the time of writing does not support signals.
     *
     * SECTION("test_assert_false")
     * {
     *  vine_assert(0);
     *  ck_abort_msg("Should've aborted...");
     * }
     */

    SECTION("test_assert_true")
    {
        vine_assert(1);
    }

    vine_final_exit(vpipe);

    test_common_teardown();
}
