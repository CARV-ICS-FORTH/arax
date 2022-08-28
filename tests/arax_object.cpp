#include "core/arax_object.h"
#include "core/arax_data.h"
#include "testing.h"

arax_pipe_s *vpipe;
arax_object_repo_s *repo;

typedef arax_object_s * (object_init_fn)(arax_pipe_s *vpipe, int over_allocate);

arax_object_s* arax_accel_s_init(arax_pipe_s *vpipe, int over_allocate)
{
    return (arax_object_s *) arax_accel_init(vpipe, "Obj", ANY, 10, 10);
}

arax_object_s* arax_vaccel_s_init(arax_pipe_s *vpipe, int over_allocate)
{
    return (arax_object_s *) arax_vaccel_init(vpipe, "Obj", ANY, 0);
}

arax_object_s* arax_proc_s_init(arax_pipe_s *vpipe, int over_allocate)
{
    return (arax_object_s *) arax_proc_init(&(vpipe->objs), "Obj");
}

arax_object_s* arax_data_s_init(arax_pipe_s *vpipe, int over_allocate)
{
    arax_object_s *obj = (arax_object_s *) arax_data_init(vpipe, over_allocate);

    arax_object_rename(obj, "Obj");
    return obj;
}

arax_object_s* arax_task_msg_s_init(arax_pipe_s *vpipe, int over_allocate)
{
    arax_object_s *obj = (arax_object_s *) arax_task_alloc(vpipe, 0, 0, over_allocate, 0, 0, 0, 0);

    arax_object_rename(obj, "Obj");
    return obj;
}

object_init_fn *initializer[ARAX_TYPE_COUNT] = {
    arax_accel_s_init,
    arax_vaccel_s_init,
    arax_proc_s_init,
    arax_data_s_init,
    arax_task_msg_s_init
};

TEST_CASE("Arax Object Tests")
{
    test_common_setup();

    int fd = test_open_config();

    const char *config = test_create_config(0x1000000);

    write(fd, config, strlen(config) );

    close(fd);

    vpipe = arax_first_init();

    for (int over_allocate = 0; over_allocate <= 1024; over_allocate += 1024) {
        for (int type = 0; type < ARAX_TYPE_COUNT; type++) {
            DYNAMIC_SECTION("test_arax_object_leak Type:" << type << " Overallocate: " << over_allocate)
            {
                arax_object_s *obj;
                arax_object_type_e otype = (arax_object_type_e) type;

                if (type == ARAX_TYPE_TASK)
                    return;

                repo = &(vpipe->objs);

                obj = initializer[type](vpipe, over_allocate);
                REQUIRE(obj);
                REQUIRE(arax_object_refs(obj) == 1);
                REQUIRE(get_object_count(repo, otype) == 1);
                REQUIRE(std::string(obj->name) == "Obj");
                arax_object_rename(obj, "Obj2");
                REQUIRE(std::string(obj->name) == "Obj2");
                arax_object_ref_dec(obj);
                REQUIRE(get_object_count(repo, otype) == 0);
            }
        }
    }

    arax_final_exit(vpipe);

    test_common_teardown();
}

TEST_CASE("arax_object_type_to_str Tests")
{
    for (int type = -ARAX_TYPE_COUNT; type < ARAX_TYPE_COUNT * 2; type++) {
        switch (type) {
            case 0 ... ARAX_TYPE_COUNT - 1:
                REQUIRE(arax_object_type_to_str((arax_object_type_e) type));
                break;
            default:
                REQUIRE(arax_object_type_to_str((arax_object_type_e) type) == 0);
        }
    }
}
