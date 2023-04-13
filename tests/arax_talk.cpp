#include "testing.h"
#include "arax_pipe.h"
#include "core/arax_data.h"

void setup()
{ }

void teardown()
{ }

TEST_CASE("arax tests")
{
    const char *config = test_create_config(0x1000000);

    test_common_setup(config);

    arax_pipe_s *vpipe = arax_first_init();

    arax_controller_init_done();

    SECTION("test_in_out"){
        arax_pipe_s *vpipe2 = arax_init();

        REQUIRE(!!vpipe2);
        REQUIRE(vpipe == vpipe2);

        REQUIRE(arax_pipe_mmap_address(vpipe) == vpipe);
        REQUIRE(arax_pipe_have_to_mmap(vpipe, system_process_id()) == 0);
        arax_exit();
        REQUIRE(arax_pipe_mmap_address(vpipe) == vpipe);
        REQUIRE(arax_pipe_have_to_mmap(vpipe, system_process_id()) == 0);
    }

    SECTION("test_revision")
    {
        char *rev;

        /**
         * Break sha intentionally
         */
        rev = (char *) arax_pipe_get_revision(vpipe);
        REQUIRE(rev);
        REQUIRE(std::string(rev) == ARAX_GIT_REV);
    }

    for (int etype = 0; etype < ARAX_ACCEL_TYPES; etype++) {
        arax_accel_type_e vtype = (arax_accel_type_e) etype;
        if (etype) {
            DYNAMIC_SECTION("test_single_proc" << arax_accel_type_to_str(vtype))
            {
                arax_proc_s *proc;
                char pd[] = "TEST_FUNCTOR";
                int cnt;

                proc = create_proc(vpipe, "TEST_PROC");

                REQUIRE(arax_proc_register("TEST_PROC") == proc);

                REQUIRE(arax_proc_can_run_at(proc, vtype) == 0);

                arax_proc_set_functor(proc, vtype, (AraxFunctor *) pd);

                REQUIRE(arax_proc_can_run_at(proc, vtype) == 1);

                AraxFunctor *vf = arax_proc_get_functor(proc, vtype);

                REQUIRE(vf == (AraxFunctor *) pd);

                for (cnt = !ANY; cnt < ARAX_ACCEL_TYPES; cnt++) {
                    if (cnt == vtype) {
                        REQUIRE(arax_proc_get_functor(proc, (arax_accel_type_e) cnt) != NULL);
                    } else {
                        REQUIRE(!arax_proc_get_functor(proc, (arax_accel_type_e) cnt) );
                    }
                    REQUIRE(arax_proc_can_run_at(proc, (arax_accel_type_e) cnt) == (cnt == vtype) );
                }

                REQUIRE(!arax_proc_put(proc) );

                REQUIRE(!arax_proc_get("TEST_PROC") );
            } /* START_TEST */
        }

        DYNAMIC_SECTION("test_task_issue_and_wait_v1" << arax_accel_type_to_str(vtype))
        {
            arax_vaccel_s *accel;

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PROC) == 0);

            arax_proc_s *issue_proc = create_proc(vpipe, "issue_proc");

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PROC) == 1);

            accel = (arax_vaccel_s *) arax_accel_acquire_type(vtype);

            REQUIRE(arax_object_refs((arax_object_s *) accel) == 1);

            void *task_handler_state = handle_n_tasks(UTILS_QUEUE_CAPACITY * 2, vtype);

            for (int cnt = 0; cnt < UTILS_QUEUE_CAPACITY * 2; cnt++) {
                arax_task *task = arax_task_issue(accel, issue_proc, 0, 0, 0, 0, 0, 0);

                REQUIRE(arax_object_refs((arax_object_s *) accel) == 2);

                arax_task_wait_done((arax_task_msg_s *) task);

                REQUIRE(arax_task_stat(task, 0) == task_completed);

                arax_task_free(task);

                REQUIRE(arax_object_refs((arax_object_s *) accel) == 1);
            }

            // Normally scheduler would set phys to something valid.
            REQUIRE(accel->phys);

            arax_accel_s *phys = accel->phys;

            // The physical accelerator should only be referenced by accel
            REQUIRE(arax_object_refs((arax_object_s *) phys) == 1);

            REQUIRE(arax_object_refs((arax_object_s *) accel) == 1);

            arax_accel_release((arax_accel **) &accel);

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 1);

            REQUIRE(handled_tasks(task_handler_state));

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 0);

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PROC) == 1);

            arax_proc_put(issue_proc);

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PROC) == 0);
        } /* START_TEST */

        DYNAMIC_SECTION("test_task_issue_sync" << arax_accel_type_to_str(vtype))
        {
            arax_vaccel_s *accel;

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PROC) == 0);

            arax_proc_s *issue_proc = create_proc(vpipe, "issue_proc");

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PROC) == 1);

            accel = (arax_vaccel_s *) arax_accel_acquire_type(vtype);

            REQUIRE(arax_object_refs((arax_object_s *) accel) == 1);

            void *task_handler_state = handle_n_tasks(UTILS_QUEUE_CAPACITY * 2, vtype);

            for (int cnt = 0; cnt < UTILS_QUEUE_CAPACITY * 2; cnt++) {
                REQUIRE(arax_task_issue_sync(accel, issue_proc, 0, 0, 0, 0, 0, 0) == task_completed);

                REQUIRE(arax_object_refs((arax_object_s *) accel) == 1);
            }

            // Normally scheduler would set phys to something valid.
            REQUIRE(accel->phys);

            arax_accel_s *phys = accel->phys;

            // The physical accelerator should only be referenced by accel
            REQUIRE(arax_object_refs((arax_object_s *) phys) == 1);

            REQUIRE(arax_object_refs((arax_object_s *) accel) == 1);

            arax_accel_release((arax_accel **) &accel);

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 1);

            REQUIRE(handled_tasks(task_handler_state));

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 0);

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PROC) == 1);

            arax_proc_put(issue_proc);

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PROC) == 0);
        } /* START_TEST */
    }
    for (arax_accel_type_e type = ANY; type < ARAX_ACCEL_TYPES + 2; type = (arax_accel_type_e) (((int) type) + 1)) {
        DYNAMIC_SECTION("test_type_strings" << arax_accel_type_to_str(type))
        {
            switch (type) {
                case 0 ... ARAX_ACCEL_TYPES:
                    REQUIRE(type == arax_accel_type_from_str(arax_accel_type_to_str(type)));
                    break;
                case ARAX_ACCEL_TYPES + 1:
                    REQUIRE(arax_accel_type_from_str("NotRealyAType") == ARAX_ACCEL_TYPES);
                    REQUIRE(!arax_accel_type_to_str(ARAX_ACCEL_TYPES));
                    REQUIRE(!arax_accel_type_to_str((arax_accel_type_e) (ARAX_ACCEL_TYPES + 1)));
                    break;
            }
        }
    }

    SECTION("test_empty_task")
    {
        arax_pipe_s *vpipe = arax_first_init();

        arax_proc_s *proc = create_proc(vpipe, "test_proc");

        arax_vaccel_s *vac = (arax_vaccel_s *) arax_accel_acquire_type(ANY);

        arax_task *task = arax_task_alloc(vpipe, vac, proc, 0, 0, 0, 0, 0);

        arax_task_mark_done((arax_task_msg_s *) task, task_completed);

        arax_task_wait(task);

        arax_task_free(task);

        arax_object_ref_dec(&(proc->obj));

        arax_final_exit(vpipe);
    }

    /**
     * This test can not be potred to Catch2 as it has to abort(signal 6).
     * Catch2 at the time of writing does not support signals.
     *
     * SECTION("test_assert_false")
     * {
     *  arax_assert(0);
     *  ck_abort_msg("Should've aborted...");
     * }
     */

    SECTION("test_assert_true")
    {
        arax_assert(1);
    }

    arax_final_exit(vpipe);

    test_common_teardown();
}
