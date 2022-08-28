#include "testing.h"
#include "arax_pipe.h"
#include "core/arax_data.h"
#include "core/arax_data_private.h"
#include "core/arax_ptr.h"

TEST_CASE("Data Tests")
{
    test_common_setup();

    int fd = test_open_config();

    const char *config = test_create_config(0x1000000);

    write(fd, config, strlen(config) );

    close(fd);

    arax_pipe_s *vpipe = arax_first_init();

    arax_controller_init_done();

    int prev_size = ARAX_BUFF_ALLOC_SIZE(0, 1) - 1;

    for (int size = 0; size < 256; size++) {
        DYNAMIC_SECTION("ARAX_BUFF_ALLOC_SIZE #" << size)
        {
            int new_size = ARAX_BUFF_ALLOC_SIZE(size, 1);

            REQUIRE(new_size > prev_size);
            REQUIRE(new_size > 0);
            prev_size = new_size;
        }
    }

    SECTION("data_leak")
    {
        // arax_data_deref causes buffer allocation in shm, ensure throttle aggrees
        size_t initial_space = arax_throttle_get_available_size(&(vpipe->throttle));
        size_t capacity      = arax_throttle_get_total_size(&(vpipe->throttle));

        REQUIRE(initial_space <= capacity); // Space should always be <= capacity

        arax_data_s *data = arax_data_init(vpipe, 1);

        arax_data_free(data);

        size_t final_space = arax_throttle_get_available_size(&(vpipe->throttle));

        REQUIRE(initial_space == final_space); // Leak in metadata

        data = arax_data_init(vpipe, 1);
        arax_data_deref(data);
        arax_data_free(data);

        final_space = arax_throttle_get_available_size(&(vpipe->throttle));

        REQUIRE(initial_space == final_space); // Leak in metadata
    }

    for (int size = 0; size < 8; size++) {
        for (int align = 1; align < 256; align *= 2) {
            DYNAMIC_SECTION("alloc_data_alligned Size: " << size << " Align: " << align)
            {
                arax_data *data = arax_data_init_aligned(vpipe, size, align);

                arax_data_stat(data);

                arax_data_check_flags((arax_data_s *) data);

                REQUIRE(data != NULL);

                REQUIRE(arax_ptr_valid(data));

                REQUIRE(arax_data_deref(data) != NULL);

                REQUIRE((size_t) arax_data_deref(data) % align == 0);

                REQUIRE(arax_ptr_valid(((char *) arax_data_deref(data)) + size));

                REQUIRE(arax_data_ref(arax_data_deref(data)) == data);

                REQUIRE(arax_data_size(data) == size);

                arax_data_free(data);
            } /* DYNAMIC_SECTION */
        }
    }

    for (int offset = -24; offset < 25; offset++) {
        DYNAMIC_SECTION("data_ref_offset Offset:" << offset)
        {
            arax_data_s *data = arax_data_init(vpipe, 16);
            void *start       = arax_data_deref(data);
            void *end         = ((char *) arax_data_deref(data)) + arax_data_size(data);
            void *test_ptr    = ((char *) start) + offset;

            if (test_ptr >= start && test_ptr < end) { // Should be inside buffer
                REQUIRE(arax_data_ref_offset(vpipe, test_ptr) == data);
            } else { // 'Outside' of buffer range
                REQUIRE(arax_data_ref_offset(vpipe, test_ptr) == 0);
            }

            arax_data_free(data);
        }
    }

    for (size_t size = 0; size < 3; size++) {
        DYNAMIC_SECTION("alloc_data")
        {
            // Physical accel
            arax_accel_s *phys = arax_accel_init(vpipe, "FakePhysAccel", ANY, 100, 10000);

            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_PHYS_ACCEL) == 1);

            // Virtual accels - assigned to phys
            arax_vaccel_s *vac_1 = (arax_vaccel_s *) arax_accel_acquire_type(ANY);

            REQUIRE(vac_1);
            arax_accel_add_vaccel(phys, vac_1);
            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_VIRT_ACCEL) == 2);

            arax_vaccel_s *vac_2 = (arax_vaccel_s *) arax_accel_acquire_type(ANY);

            REQUIRE(vac_2);
            arax_accel_add_vaccel(phys, vac_2);
            REQUIRE(get_object_count(&(vpipe->objs), ARAX_TYPE_VIRT_ACCEL) == 3);

            arax_data *data         = ARAX_BUFFER(size);
            arax_object_s *data_obj = &(((arax_data_s *) data)->obj);

            REQUIRE(arax_object_refs(data_obj) == 1);

            REQUIRE(data != NULL);

            REQUIRE(arax_data_has_remote(data) == 0);

            REQUIRE(arax_data_deref(data) == ((arax_data_s *) data)->buffer);

            REQUIRE(arax_data_deref(data) != NULL);

            REQUIRE(arax_data_ref(arax_data_deref(data)) == data);

            arax_data_check_flags((arax_data_s *) data);

            REQUIRE(arax_data_size(data) == size);

            // Just call these functions - they should not crash
            // Eventually add more thorough tests.

            REQUIRE(arax_object_refs(data_obj) == 1);
            REQUIRE(((arax_data_s *) data)->accel == 0);
            arax_data_arg_init((arax_data_s *) data, vac_1);
            REQUIRE(vac_1 == ((arax_data_s *) data)->accel);
            REQUIRE(arax_object_refs(data_obj) == 1);
            arax_data_input_init((arax_data_s *) data, vac_1);
            REQUIRE(vac_1 == ((arax_data_s *) data)->accel);
            REQUIRE(arax_object_refs(data_obj) == 2);
            arax_data_output_init((arax_data_s *) data, vac_1);
            REQUIRE(vac_1 == ((arax_data_s *) data)->accel);
            REQUIRE(arax_object_refs(data_obj) == 3);
            arax_data_memcpy(0, (arax_data_s *) data, (arax_data_s *) data, 0);
            REQUIRE(vac_1 == ((arax_data_s *) data)->accel);

            // Repeat tests , with different vac, but pointing to same phys

            REQUIRE(arax_object_refs(data_obj) == 3);
            REQUIRE(vac_1 == ((arax_data_s *) data)->accel);
            arax_data_arg_init((arax_data_s *) data, vac_2);
            REQUIRE(vac_2 == ((arax_data_s *) data)->accel);
            REQUIRE(arax_object_refs(data_obj) == 3);
            arax_data_input_init((arax_data_s *) data, vac_2);
            REQUIRE(vac_2 == ((arax_data_s *) data)->accel);
            REQUIRE(arax_object_refs(data_obj) == 4);
            arax_data_output_init((arax_data_s *) data, vac_2);
            REQUIRE(vac_2 == ((arax_data_s *) data)->accel);
            REQUIRE(arax_object_refs(data_obj) == 5);
            arax_data_memcpy(0, (arax_data_s *) data, (arax_data_s *) data, 0);
            REQUIRE(vac_2 == ((arax_data_s *) data)->accel);

            unsigned int i = 5;

            REQUIRE(i == arax_object_refs(data_obj)); // We did 5 arax_data_*_init calls
            for (; i > 0; i--)
                arax_data_free(data);

            arax_accel_release((arax_accel **) &vac_1);

            arax_accel_release((arax_accel **) &vac_2);

            arax_accel_release((arax_accel **) &phys);
        } /* DYNAMIC_SECTION */
    }

    arax_final_exit(vpipe);

    test_common_teardown();
}
