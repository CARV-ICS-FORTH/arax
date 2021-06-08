#include "utils/bitmap.h"
#include "testing.h"

void setup()
{ }

void teardown(){ }

static const int bitmap_size = 512;

void alloc_bits_check(size_t size, utils_bitmap_s *bmp)
{
    size_t count = utils_bitmap_count_allocated(bmp);

    ck_assert_int_eq(size, count);
    ck_assert_int_eq(utils_bitmap_used(bmp), count);
}

START_TEST(test_bitmap_init)
{
    size_t bits     = _i;
    size_t bmp_size = UTILS_BITMAP_CALC_BYTES(bits);
    uint8_t *buff   = malloc(bmp_size + 1);

    memset(buff, 0xFF, bmp_size + 1);


    utils_bitmap_s *bmp = utils_bitmap_init(buff, bits);

    ck_assert_ptr_eq(bmp, buff);

    ck_assert_int_eq(utils_bitmap_size(bmp), bits);
    ck_assert_int_eq(utils_bitmap_used(bmp), 0);
    ck_assert_int_eq(utils_bitmap_free(bmp), bits);

    ck_assert_int_eq(0xFF, buff[bmp_size]);
    free(buff);
}
END_TEST START_TEST(test_bitmap_single_bit)
{
    size_t bmp_size = UTILS_BITMAP_CALC_BYTES(bitmap_size);
    uint8_t *buff   = malloc(bmp_size + 1);

    memset(buff, 0xFF, bmp_size + 1);

    utils_bitmap_s *bmp = utils_bitmap_init(buff, bitmap_size);

    for (int b = 0; b < bitmap_size; b++) { // Allocate all the bits
        ck_assert_int_eq(b, utils_bitmap_used(bmp));
        size_t bit = utils_bitmap_alloc_bits(bmp, 1);
        ck_assert_int_ne(bit, BITMAP_NOT_FOUND);
        alloc_bits_check(b + 1, bmp);
    }

    ck_assert_int_eq(0, utils_bitmap_free(bmp));
    // Bitmap is full, next alloc should fail
    size_t bit = utils_bitmap_alloc_bits(bmp, 1);

    ck_assert_int_eq(bit, BITMAP_NOT_FOUND);

    for (int b = 0; b < bitmap_size; b++) { // Free all the bits
        ck_assert_int_eq(b, utils_bitmap_free(bmp));
        utils_bitmap_free_bits(bmp, b, 1);
        alloc_bits_check(bitmap_size - (b + 1), bmp);
    }

    alloc_bits_check(0, bmp);

    for (int b = 0; b < bitmap_size; b++) { // Allocate all the bits
        ck_assert_int_eq(b, utils_bitmap_used(bmp));
        size_t bit = utils_bitmap_alloc_bits(bmp, 1);
        ck_assert_int_ne(bit, BITMAP_NOT_FOUND);
        alloc_bits_check(b + 1, bmp);
    }

    for (int b = 0; b < bitmap_size / 2; b++) { // Free first half of bits
        ck_assert_int_eq(b, utils_bitmap_free(bmp));
        utils_bitmap_free_bits(bmp, b, 1);
        alloc_bits_check(bitmap_size - (b + 1), bmp);
    }

    for (int b = 0; b < bitmap_size / 2; b++) { // Allocate first half of bits
        ck_assert_int_eq(bitmap_size / 2 + b, utils_bitmap_used(bmp));
        size_t bit = utils_bitmap_alloc_bits(bmp, 1);
        ck_assert_int_ne(bit, BITMAP_NOT_FOUND);
        alloc_bits_check(bitmap_size / 2 + b + 1, bmp);
    }

    for (int b = 0; b < bitmap_size / 2; b++) { // Free second half of bits
        ck_assert_int_eq(b, utils_bitmap_free(bmp));
        utils_bitmap_free_bits(bmp, b + bitmap_size / 2, 1);
        alloc_bits_check(bitmap_size - (b + 1), bmp);
    }

    for (int b = 0; b < bitmap_size / 2; b++) { // Allocate second half of bits
        ck_assert_int_eq(bitmap_size / 2 + b, utils_bitmap_used(bmp));
        size_t bit = utils_bitmap_alloc_bits(bmp, 1);
        ck_assert_int_eq(bit, b + bitmap_size / 2);
        alloc_bits_check(bitmap_size / 2 + (b + 1), bmp);
    }

    for (int b = 0; b < bitmap_size; b += 2) { // Free even bits
        ck_assert_int_eq(b / 2, utils_bitmap_free(bmp));
        utils_bitmap_free_bits(bmp, b, 1);
        alloc_bits_check(bitmap_size - (b / 2 + 1), bmp);
    }

    ck_assert_int_eq(bitmap_size / 2, utils_bitmap_free(bmp));

    for (int b = 0; b < bitmap_size; b += 2) { // Allocate even bits
        ck_assert_int_eq(bitmap_size / 2 + b / 2, utils_bitmap_used(bmp));
        size_t bit = utils_bitmap_alloc_bits(bmp, 1);
        ck_assert_int_ne(bit, BITMAP_NOT_FOUND);
        alloc_bits_check(bitmap_size / 2 + (b / 2 + 1), bmp);
    }

    ck_assert_int_eq(0, utils_bitmap_free(bmp));

    for (int b = 1; b < bitmap_size; b += 2) { // Free odd bits
        ck_assert_int_eq(b / 2, utils_bitmap_free(bmp));
        utils_bitmap_free_bits(bmp, b, 1);
        alloc_bits_check(bitmap_size - (b / 2 + 1), bmp);
    }

    ck_assert_int_eq(bitmap_size / 2, utils_bitmap_free(bmp));

    for (int b = 1; b < bitmap_size; b += 2) { // Allocate even bits
        ck_assert_int_eq(bitmap_size / 2 + b / 2, utils_bitmap_used(bmp));
        size_t bit = utils_bitmap_alloc_bits(bmp, 1);
        ck_assert_int_ne(bit, BITMAP_NOT_FOUND);
        alloc_bits_check(bitmap_size / 2 + (b / 2 + 1), bmp);
    }

    ck_assert_int_eq(0, utils_bitmap_free(bmp));

    srand(0); // TODO: make own rand to ensure determinism

    // Fuzzy test it

    int tests = bitmap_size * 100;

    while (tests--) {
        int bit = rand() % bitmap_size;

        utils_bitmap_free_bits(bmp, bit, 1);
        ck_assert_int_eq(bit, utils_bitmap_alloc_bits(bmp, 1));
    }

    ck_assert_int_eq(0xFF, buff[bmp_size]);
    free(buff);
} /* START_TEST */

END_TEST START_TEST(test_bitmap_multi_bit)
{
    size_t alloc_size = _i;
    size_t bmp_size   = UTILS_BITMAP_CALC_BYTES(bitmap_size);
    uint8_t *buff     = malloc(bmp_size + 1);

    memset(buff, 0xFF, bmp_size + 1);

    utils_bitmap_s *bmp = utils_bitmap_init(buff, bitmap_size);

    for (int a = 0; a < (bitmap_size / alloc_size); a++) { // Allocate all you can, sequentialy
        ck_assert_int_eq(a * alloc_size, utils_bitmap_used(bmp));
        utils_bitmap_alloc_bits(bmp, alloc_size);
        alloc_bits_check((a + 1) * alloc_size, bmp);
    }

    size_t bit = utils_bitmap_alloc_bits(bmp, alloc_size);

    ck_assert_int_eq(bit, BITMAP_NOT_FOUND);

    int remainder = utils_bitmap_free(bmp);

    ck_assert_int_lt(remainder, alloc_size);

    if (remainder) { // Allocate remainder
        bit = utils_bitmap_alloc_bits(bmp, remainder);
        ck_assert_int_ne(bit, BITMAP_NOT_FOUND);
    } else { // No remainder
        bit = utils_bitmap_alloc_bits(bmp, 1);
        ck_assert_int_eq(bit, BITMAP_NOT_FOUND);
    }

    for (int a = 0; a < (bitmap_size / alloc_size); a++) { // Free all you can, sequentialy
        ck_assert_int_eq(a * alloc_size, utils_bitmap_free(bmp));
        utils_bitmap_free_bits(bmp, alloc_size * a, alloc_size);
        alloc_bits_check(bitmap_size - (a + 1) * alloc_size, bmp);
    }

    if (remainder) { // Free remainder
        alloc_bits_check(remainder, bmp);
        utils_bitmap_free_bits(bmp, bitmap_size - remainder, remainder);
        alloc_bits_check(0, bmp);
    } else { // No remainder
        alloc_bits_check(0, bmp);
    }

    // Allocate all
    ck_assert_int_eq(utils_bitmap_alloc_bits(bmp, bitmap_size), 0);

    alloc_bits_check(bitmap_size, bmp); // All full

    // Free all
    utils_bitmap_free_bits(bmp, 0, bitmap_size);

    alloc_bits_check(0, bmp); // All empty

    // Fuzzy Test

    int tests = 10 * bitmap_size;

    srand(0);

    while (tests--) {
        if (utils_bitmap_free(bmp) == 0) { // All full
            do{
                size_t start = rand() % bitmap_size;
                utils_bitmap_free_bits(bmp, start, 1);
            } while(utils_bitmap_free(bmp) < bitmap_size / 2);
        } else {
            size_t used = utils_bitmap_used(bmp);
            size_t span = rand() % utils_bitmap_free(bmp);
            while (utils_bitmap_alloc_bits(bmp, span) == BITMAP_NOT_FOUND)
                span--;
            size_t count = utils_bitmap_count_allocated(bmp);
            ck_assert_int_eq(used + span, count);
            ck_assert_int_eq(utils_bitmap_used(bmp), count);
        }
    }

    ck_assert_int_eq(0xFF, buff[bmp_size]);
    free(buff);
} /* START_TEST */

END_TEST START_TEST(test_bitmap_full_alloc)
{
    size_t alloc_size = _i;
    size_t bmp_size   = UTILS_BITMAP_CALC_BYTES(bitmap_size);
    uint8_t *buff     = malloc(bmp_size + 1);

    memset(buff, 0xFF, bmp_size + 1);

    utils_bitmap_s *bmp = utils_bitmap_init(buff, bitmap_size);

    // First alloc should start at 0
    ck_assert_int_eq(utils_bitmap_alloc_bits(bmp, bitmap_size - alloc_size), 0);
    alloc_bits_check(bitmap_size - alloc_size, bmp);

    // Second alloc should end where the first ended
    ck_assert_int_eq(utils_bitmap_alloc_bits(bmp, alloc_size), bitmap_size - alloc_size);
    alloc_bits_check(bitmap_size, bmp);

    utils_bitmap_free_bits(bmp, bitmap_size - alloc_size, alloc_size);

    alloc_bits_check(bitmap_size - alloc_size, bmp);

    utils_bitmap_free_bits(bmp, 0, bitmap_size - alloc_size);

    alloc_bits_check(0, bmp);

    ck_assert_int_eq(0xFF, buff[bmp_size]);
    free(buff);
}

END_TEST

Suite* suite_init()
{
    Suite *s;
    TCase *tc_single;

    s         = suite_create("Bitmap");
    tc_single = tcase_create("Single");
    tcase_add_unchecked_fixture(tc_single, setup, teardown);

    tcase_add_loop_test(tc_single, test_bitmap_init, 1, 128);
    tcase_add_test(tc_single, test_bitmap_single_bit);
    tcase_add_loop_test(tc_single, test_bitmap_multi_bit, 1, bitmap_size);
    tcase_add_loop_test(tc_single, test_bitmap_full_alloc, 1, bitmap_size);

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
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_NORMAL);
    failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (failed) ? EXIT_FAILURE : EXIT_SUCCESS;
}
