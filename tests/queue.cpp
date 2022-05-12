#include <stddef.h>
#include <stdint.h>
#include "utils/queue.h"
#include "testing.h"

#define CONSUMER_POP_COUNT 128
/* Each consumer consumes CONSUMER_POP_COUNT items */
void* consumer(void *data)
{
    utils_queue_s **queue = (utils_queue_s **) data;
    int c;
    int ret;
    int sum = 0;

    while (!*queue);  // Wait for sync signal

    for (c = 0; c < CONSUMER_POP_COUNT; c++) {
        do{
            ret = (size_t) utils_queue_pop(*queue);
        }while (!ret);
        sum += ret;
    }
    return (void *) (size_t) sum;
}

TEST_CASE("Queue Tests")
{
    const size_t BUFF_SIZE = sizeof(utils_queue_s) + 2;

    char _buff[BUFF_SIZE + 64];

    char *buff = _buff + (63 - (((size_t) _buff) & 63));

    utils_queue_s *queue;
    char FULL_OF_FF = 0xFF;

    memset(buff, FULL_OF_FF, BUFF_SIZE);
    queue = utils_queue_init(buff + 1);
    REQUIRE(queue);
    REQUIRE(utils_queue_used_slots(queue) == 0);
    REQUIRE(utils_queue_peek(queue) == 0);

    SECTION("test_queue_push_pop")
    {
        int c = UTILS_QUEUE_CAPACITY;

        while (c) {
            REQUIRE(utils_queue_push(queue, (void *) (size_t) c) == (void *) (size_t) c);
            c--;
        }

        REQUIRE(!utils_queue_push(queue, (void *) (((size_t) c) + 1)) );

        c = utils_queue_used_slots(queue);
        while (c) {
            REQUIRE(utils_queue_peek(queue) == (void *) (size_t) c);
            REQUIRE(utils_queue_pop(queue) == (void *) (size_t) c);
            c--;
        }

        REQUIRE(utils_queue_used_slots(queue) == 0);
        REQUIRE(utils_queue_pop(queue) == 0);
    }

    /* Check that the circular buffer works OK */
    SECTION("test_queue_circulation")
    {
        /* Fill half buffer */
        int c = UTILS_QUEUE_CAPACITY / 2;

        while (c) {
            REQUIRE(utils_queue_push(queue, (void *) (size_t) c) == (void *) (size_t) c);
            c--;
        }

        /* Empty it */
        c = utils_queue_used_slots(queue);
        while (c) {
            REQUIRE(utils_queue_pop(queue) == (void *) (size_t) c);
            c--;
        }

        /* Now fill it and it should wrap around */
        c = UTILS_QUEUE_CAPACITY;
        while (c) {
            REQUIRE(utils_queue_push(queue, (void *) (size_t) c) == (void *) (size_t) c);
            c--;
        }

        REQUIRE(!utils_queue_push(queue, (void *) (((size_t) c) + 1)) );

        c = utils_queue_used_slots(queue);
        while (c) {
            REQUIRE(utils_queue_pop(queue) == (void *) (size_t) c);
            c--;
        }

        REQUIRE(utils_queue_used_slots(queue) == 0);
        REQUIRE(utils_queue_pop(queue) == 0);
    } /* START_TEST */


    /* Check that uint16_t wrap around does not cause any issues */
    SECTION("test_queue_indices_circulation")
    {
        int c = UTILS_QUEUE_CAPACITY;

        while (c) {
            REQUIRE(utils_queue_push(queue, (void *) (size_t) c) == (void *) (size_t) c);
            c--;
        }

        REQUIRE(utils_queue_push(queue, (void *) 0xFF) == 0);
        REQUIRE(utils_queue_used_slots(queue) == UTILS_QUEUE_CAPACITY);

        c = 2 * UINT16_MAX;
        while (c) {
            REQUIRE(utils_queue_pop(queue) != NULL);
            REQUIRE(utils_queue_push(queue, (void *) (size_t) c) == (void *) (size_t) c);
            c--;
        }

        REQUIRE(utils_queue_push(queue, (void *) 0xFF) == 0);
        REQUIRE(utils_queue_used_slots(queue) == UTILS_QUEUE_CAPACITY);
    }

    /* Test queue with one or more consumers and a single producer */
    for (int tcnt = 1; tcnt < 4; tcnt++) {
        DYNAMIC_SECTION("MCSP " << tcnt << " threads")
        {
            int c;
            int sum = 0;
            void *ret;
            pthread_t threads[tcnt];

            utils_queue_s *sync = 0;

            for (c = 0; c < tcnt; c++) {
                pthread_create(threads + c, 0, consumer, &sync);
            }

            safe_usleep(10000); // 1ms to let consumers spawn

            sync = queue; // Let consumers produce

            for (c = 1; c < CONSUMER_POP_COUNT * tcnt + 1; c++) {
                while (!utils_queue_push(queue, (void *) (size_t) c));
            }

            for (c = 0; c < tcnt; c++) {
                pthread_join(threads[c], &ret);
                sum += (size_t) ret;
            }
            c = CONSUMER_POP_COUNT * tcnt;
            REQUIRE(sum == (c * (c + 1)) / 2); // sum of series 1,2,3 * consumers
        }
    }

    REQUIRE(buff[0] == FULL_OF_FF);
    REQUIRE(buff[BUFF_SIZE - 1] == FULL_OF_FF);
}
