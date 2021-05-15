#include "core/vine_object.h"
#include "core/vine_data.h"
#include "testing.h"

vine_object_repo_s repo;
union {
    arch_alloc_s alloc;
    char         pool[8192];
} pool;

typedef void (*vine_object_dtor)(vine_object_s *obj);

void setup()
{
    arch_alloc_init(&pool.alloc, sizeof(pool));
    vine_object_repo_init(&repo, &pool.alloc);
}

void teardown()
{
    ck_assert(!vine_object_repo_exit(&repo));
    arch_alloc_exit(&pool.alloc);
}

START_TEST(test_vine_object_init_destr){ }
END_TEST

union AllObjects {
    vine_task_msg_s msg;
    vine_vaccel_s   va;
    vine_accel_s    pa;
    vine_data_s     data;
    vine_proc_s     proc;
};

START_TEST(test_vine_object_leak)
{
    int over_allocate = (_i >= VINE_TYPE_COUNT) * 1024;
    int type = _i % VINE_TYPE_COUNT;
    vine_object_s *obj;

    obj = vine_object_register(&repo, type, "Obj", sizeof(union AllObjects) + over_allocate, 1);
    ck_assert(obj);
    ck_assert_int_eq(vine_object_refs(obj), 1);
    ck_assert_int_eq(get_object_count(&repo, type), 1);
    ck_assert_str_eq(obj->name, "Obj");
    vine_object_rename(obj, "Obj2");
    ck_assert_str_eq(obj->name, "Obj2");
    ck_assert_int_eq(vine_object_repo_exit(&repo), 1);
    vine_object_ref_dec(obj);
    ck_assert_int_eq(get_object_count(&repo, type), 0);
}

END_TEST Suite* suite_init()
{
    Suite *s;
    TCase *tc_single;

    s         = suite_create("vine_object");
    tc_single = tcase_create("Single");
    tcase_add_unchecked_fixture(tc_single, setup, teardown);
    tcase_add_test(tc_single, test_vine_object_init_destr);
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
