#include "core/vine_object.h"
#include "core/vine_data.h"
#include "testing.h"

vine_pipe_s *vpipe;
vine_object_repo_s *repo;

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

typedef vine_object_s * (object_init_fn)(vine_pipe_s *vpipe, int over_allocate);

vine_object_s* vine_accel_s_init(vine_pipe_s *vpipe, int over_allocate)
{
    return (vine_object_s *) vine_accel_init(vpipe, "Obj", ANY, 10, 10);
}

vine_object_s* vine_vaccel_s_init(vine_pipe_s *vpipe, int over_allocate)
{
    return (vine_object_s *) vine_vaccel_init(vpipe, "Obj", ANY, 0);
}

vine_object_s* vine_proc_s_init(vine_pipe_s *vpipe, int over_allocate)
{
    return (vine_object_s *) vine_proc_init(&(vpipe->objs), "Obj");
}

vine_object_s* vine_data_s_init(vine_pipe_s *vpipe, int over_allocate)
{
    vine_object_s *obj = (vine_object_s *) vine_data_init(vpipe, 0, over_allocate);

    vine_object_rename(obj, "Obj");
    return obj;
}

vine_object_s* vine_task_msg_s_init(vine_pipe_s *vpipe, int over_allocate)
{
    vine_object_s *obj = (vine_object_s *) vine_task_alloc(vpipe, over_allocate, 0, 0);

    vine_object_rename(obj, "Obj");
    return obj;
}

object_init_fn *initializer[VINE_TYPE_COUNT] = {
    vine_accel_s_init,
    vine_vaccel_s_init,
    vine_proc_s_init,
    vine_data_s_init,
    vine_task_msg_s_init
};

START_TEST(test_vine_object_leak)
{
    vpipe = vine_first_init();
    repo  = &(vpipe->objs);

    int over_allocate = (_i >= VINE_TYPE_COUNT) * 1024;
    int type = _i % VINE_TYPE_COUNT;
    vine_object_s *obj;

    obj = initializer[type](vpipe, over_allocate);
    ck_assert(obj);
    ck_assert_int_eq(vine_object_refs(obj), 1);
    ck_assert_int_eq(get_object_count(repo, type), 1);
    ck_assert_str_eq(obj->name, "Obj");
    vine_object_rename(obj, "Obj2");
    ck_assert_str_eq(obj->name, "Obj2");
    vine_object_ref_dec(obj);
    ck_assert_int_eq(get_object_count(repo, type), 0);

    vine_final_exit(vpipe);
}

END_TEST Suite* suite_init()
{
    Suite *s;
    TCase *tc_single;

    s         = suite_create("vine_object");
    tc_single = tcase_create("Single");
    tcase_add_unchecked_fixture(tc_single, setup, teardown);
    tcase_add_loop_test(tc_single, test_vine_object_leak, 0,
      VINE_TYPE_COUNT * 2);
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
    // srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_NORMAL);
    failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
