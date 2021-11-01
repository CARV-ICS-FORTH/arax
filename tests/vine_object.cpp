#include "core/vine_object.h"
#include "core/vine_data.h"
#include "testing.h"

vine_pipe_s *vpipe;
vine_object_repo_s *repo;

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
    vine_object_s *obj = (vine_object_s *) vine_task_alloc(vpipe, 0, 0, over_allocate, 0, 0);

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

TEST_CASE("Vine Object Tests")
{
    test_common_setup();

    int fd = test_open_config();

    const char *config = test_create_config(0x1000000);

    write(fd, config, strlen(config) );

    close(fd);

    vpipe = vine_first_init();

    for (int over_allocate = 0; over_allocate <= 1024; over_allocate += 1024) {
        for (int type = 0; type < VINE_TYPE_COUNT; type++) {
            DYNAMIC_SECTION("test_vine_object_leak Type:" << type << " Overallocate: " << over_allocate)
            {
                vine_object_s *obj;
                vine_object_type_e otype = (vine_object_type_e) type;

                if (type == VINE_TYPE_TASK)
                    return;

                repo = &(vpipe->objs);

                obj = initializer[type](vpipe, over_allocate);
                REQUIRE(obj);
                REQUIRE(vine_object_refs(obj) == 1);
                REQUIRE(get_object_count(repo, otype) == 1);
                REQUIRE(std::string(obj->name) == "Obj");
                vine_object_rename(obj, "Obj2");
                REQUIRE(std::string(obj->name) == "Obj2");
                vine_object_ref_dec(obj);
                REQUIRE(get_object_count(repo, otype) == 0);
            }
        }
    }

    vine_final_exit(vpipe);

    test_common_teardown();
}
