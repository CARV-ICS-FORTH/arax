#include "testing.h"
#include "vine_pipe.h"
#include "core/vine_data.h"
#include "core/vine_ptr.h"

TEST_CASE("Data Tests")
{
    test_common_setup();

    int fd = test_open_config();

    const char *config = test_create_config(0x1000000);

    write(fd, config, strlen(config) );

    close(fd);

    vine_pipe_s *vpipe = vine_first_init();

    vine_talk_controller_init_done();

    int prev_size = VINE_BUFF_ALLOC_SIZE(0, 1) - 1;

    for (int size = 0; size < 256; size++) {
        DYNAMIC_SECTION("VINE_BUFF_ALLOC_SIZE #" << size)
        {
            int new_size = VINE_BUFF_ALLOC_SIZE(size, 1);

            REQUIRE(new_size > prev_size);
            REQUIRE(new_size > 0);
            prev_size = new_size;
        }
    }

    SECTION("data_leak")
    {
        // vine_data_deref causes buffer allocation in shm, ensure throttle aggrees
        size_t initial_space = vine_throttle_get_available_size(&(vpipe->throttle));
        size_t capacity      = vine_throttle_get_total_size(&(vpipe->throttle));

        REQUIRE(initial_space <= capacity); // Space should always be <= capacity

        vine_data_s *data = vine_data_init(vpipe, 0, 1);

        vine_data_free(data);

        size_t final_space = vine_throttle_get_available_size(&(vpipe->throttle));

        REQUIRE(initial_space == final_space); // Leak in metadata

        data = vine_data_init(vpipe, 0, 1);
        vine_data_deref(data);
        vine_data_free(data);

        final_space = vine_throttle_get_available_size(&(vpipe->throttle));

        REQUIRE(initial_space == final_space); // Leak in metadata
    }

    for (int size = 0; size < 8; size++) {
        for (int align = 1; align < 256; align *= 2) {
            DYNAMIC_SECTION("alloc_data_alligned Size: " << size << " Align: " << align)
            {
                vine_data *data = vine_data_init_aligned(vpipe, 0, size, align);

                vine_data_stat(data);

                vine_data_check_flags((vine_data_s *) data);

                REQUIRE(data != NULL);

                REQUIRE(vine_ptr_valid(data));

                REQUIRE(vine_data_deref(data) != NULL);

                REQUIRE((size_t) vine_data_deref(data) % align == 0);

                REQUIRE(vine_ptr_valid(((char *) vine_data_deref(data)) + size));

                REQUIRE(vine_data_ref(vine_data_deref(data)) == data);

                REQUIRE(vine_data_size(data) == size);

                REQUIRE(!vine_data_check_ready(vpipe, data));
                vine_data_mark_ready(vpipe, data);
                REQUIRE(vine_data_check_ready(vpipe, data));

                vine_data_free(data);
            } /* DYNAMIC_SECTION */
        }
    }

    for (int offset = -24; offset < 25; offset++) {
        DYNAMIC_SECTION("data_ref_offset Offset:" << offset)
        {
            vine_data_s *data = vine_data_init(vpipe, 0, 16);
            void *start       = vine_data_deref(data);
            void *end         = ((char *) vine_data_deref(data)) + vine_data_size(data);
            void *test_ptr    = ((char *) start) + offset;

            if (test_ptr >= start && test_ptr < end) { // Should be inside buffer
                REQUIRE(vine_data_ref_offset(vpipe, test_ptr) == data);
            } else { // 'Outside' of buffer range
                REQUIRE(vine_data_ref_offset(vpipe, test_ptr) == 0);
            }

            vine_data_free(data);
        }
    }

    for (size_t size = 0; size < 3; size++) {
        DYNAMIC_SECTION("alloc_data")
        {
            // Physical accel
            vine_accel_s *phys = vine_accel_init(vpipe, "FakePhysAccel", ANY, 100, 10000);

            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_PHYS_ACCEL) == 1);

            // Virtual accels - assigned to phys
            vine_vaccel_s *vac_1 = (vine_vaccel_s *) vine_accel_acquire_type(ANY);

            REQUIRE(vac_1);
            vine_accel_add_vaccel(phys, vac_1);
            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL) == 1);

            vine_vaccel_s *vac_2 = (vine_vaccel_s *) vine_accel_acquire_type(ANY);

            REQUIRE(vac_2);
            vine_accel_add_vaccel(phys, vac_2);
            REQUIRE(get_object_count(&(vpipe->objs), VINE_TYPE_VIRT_ACCEL) == 2);

            vine_data *data         = VINE_BUFFER(0, size);
            vine_object_s *data_obj = &(((vine_data_s *) data)->obj);

            REQUIRE(vine_object_refs(data_obj) == 1);

            REQUIRE(data != NULL);

            REQUIRE(vine_data_has_remote(data) == 0);

            REQUIRE(vine_data_deref(data) == ((vine_data_s *) data)->buffer);

            REQUIRE(vine_data_deref(data) != NULL);

            REQUIRE(vine_data_ref(vine_data_deref(data)) == data);

            vine_data_check_flags((vine_data_s *) data);

            REQUIRE(vine_data_size(data) == size);

            REQUIRE_FALSE(vine_data_check_ready(vpipe, data));
            vine_data_mark_ready(vpipe, data);
            REQUIRE(vine_data_check_ready(vpipe, data));

            // Just call these functions - they should not crash
            // Eventually add more thorough tests.

            REQUIRE(vine_object_refs(data_obj) == 1);
            REQUIRE(((vine_data_s *) data)->accel == 0);
            vine_data_arg_init((vine_data_s *) data, vac_1);
            REQUIRE(vac_1 == ((vine_data_s *) data)->accel);
            REQUIRE(vine_object_refs(data_obj) == 1);
            vine_data_input_init((vine_data_s *) data, vac_1);
            REQUIRE(vac_1 == ((vine_data_s *) data)->accel);
            REQUIRE(vine_object_refs(data_obj) == 2);
            vine_data_output_init((vine_data_s *) data, vac_1);
            REQUIRE(vac_1 == ((vine_data_s *) data)->accel);
            REQUIRE(vine_object_refs(data_obj) == 3);
            vine_data_output_done((vine_data_s *) data);
            REQUIRE(vac_1 == ((vine_data_s *) data)->accel);
            REQUIRE(vine_object_refs(data_obj) == 3);
            vine_data_memcpy(0, (vine_data_s *) data, (vine_data_s *) data, 0);
            REQUIRE(vac_1 == ((vine_data_s *) data)->accel);

            // Repeat tests , with different vac, but pointing to same phys

            REQUIRE(vine_object_refs(data_obj) == 3);
            REQUIRE(vac_1 == ((vine_data_s *) data)->accel);
            vine_data_arg_init((vine_data_s *) data, vac_2);
            REQUIRE(vac_2 == ((vine_data_s *) data)->accel);
            REQUIRE(vine_object_refs(data_obj) == 3);
            vine_data_input_init((vine_data_s *) data, vac_2);
            REQUIRE(vac_2 == ((vine_data_s *) data)->accel);
            REQUIRE(vine_object_refs(data_obj) == 4);
            vine_data_output_init((vine_data_s *) data, vac_2);
            REQUIRE(vac_2 == ((vine_data_s *) data)->accel);
            REQUIRE(vine_object_refs(data_obj) == 5);
            vine_data_output_done((vine_data_s *) data);
            REQUIRE(vine_object_refs(data_obj) == 5);
            vine_data_memcpy(0, (vine_data_s *) data, (vine_data_s *) data, 0);
            REQUIRE(vac_2 == ((vine_data_s *) data)->accel);

            // vine_data_sync_to_remote should be a no-op when data are in remote

            vine_data_modified(data, REMT_SYNC);
            vine_data_sync_to_remote(vac_1, data, 0);

            // vine_data_sync_from_remote should be a no-op when data are in USR

            vine_data_modified(data, USER_SYNC);
            vine_data_sync_from_remote(vac_1, data, 0);

            // vine_data_sync_from_remote should be a no-op when data are in SHM (and user pointer is null)

            vine_data_modified(data, SHM_SYNC);
            vine_data_sync_from_remote(vac_1, data, 0);

            unsigned int i = 5;

            REQUIRE(i == vine_object_refs(data_obj)); // We did 5 vine_data_*_init calls
            for (; i > 0; i--)
                vine_data_free(data);

            vine_accel_release((vine_accel **) &vac_1);

            vine_accel_release((vine_accel **) &vac_2);

            vine_accel_release((vine_accel **) &phys);
        } /* DYNAMIC_SECTION */
    }

    vine_final_exit(vpipe);

    test_common_teardown();
}
