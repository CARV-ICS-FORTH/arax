#include "utils/list.h"
#include "testing.h"
#define TEST_LENGTH 100

utils_list_node_s* allocate_list_node()
{
    utils_list_node_s *node = new utils_list_node_s();

    REQUIRE(!!node);
    utils_list_node_init(node, node);
    REQUIRE(!utils_list_node_linked(node));
    return node;
}

void free_list_node(utils_list_node_s *node)
{
    REQUIRE(!!node);
    delete node;
}

TEST_CASE("list tests")
{
    utils_list_s list;

    REQUIRE(utils_list_init(&list) == &list);

    for (int test_length = 0; test_length < 100; test_length++) {
        DYNAMIC_SECTION("test_list_add_del_to_array #" << test_length)
        {
            utils_list_node_s **nodes;
            void **copy;
            utils_list_node_s *itr;
            int c;

            nodes = new utils_list_node_s *[test_length];
            copy  = new void *[test_length];

            for (c = 0; c < test_length; c++) {
                REQUIRE(list.length == c);
                nodes[c] = allocate_list_node();
                utils_list_add(&list, nodes[c]);
            }
            REQUIRE(list.length == test_length);

            c = 0;
            utils_list_for_each(list, itr){
                REQUIRE(itr == itr->owner);
                REQUIRE(itr == nodes[test_length - 1 - c]);
                REQUIRE(c < test_length);
                c++;
            }
            c = 0;
            utils_list_for_each_reverse(list, itr){
                REQUIRE(itr == itr->owner);
                REQUIRE(itr == nodes[c]);
                REQUIRE(c < test_length);
                c++;
            }

            REQUIRE(utils_list_to_array(&list, 0) == test_length);
            REQUIRE(utils_list_to_array(&list, copy) == test_length);

            for (c = 0; c < test_length; c++)
                REQUIRE(nodes[c] == copy[test_length - c - 1]);

            for (c = 0; c < test_length; c++) {
                utils_list_node_s *node = utils_list_pop_head(&list);
                REQUIRE(nodes[test_length - c - 1] == node);
                free_list_node(node);
            }
        }

        DYNAMIC_SECTION("test_list_pop_tail #" << test_length)
        {
            utils_list_node_s **nodes;
            void **copy;
            utils_list_node_s *itr;
            int c;

            nodes = new utils_list_node_s *[test_length];
            copy  = new void *[test_length];

            for (c = 0; c < test_length; c++) {
                REQUIRE(list.length == c);
                nodes[c] = allocate_list_node();
                utils_list_add(&list, nodes[c]);
            }
            REQUIRE(list.length == test_length);

            c = 0;
            utils_list_for_each(list, itr){
                REQUIRE(itr == itr->owner);
                REQUIRE(itr == nodes[test_length - 1 - c]);
                REQUIRE(c < test_length);
                c++;
            }
            c = 0;
            utils_list_for_each_reverse(list, itr){
                REQUIRE(itr == itr->owner);
                REQUIRE(itr == nodes[c]);
                REQUIRE(c < test_length);
                c++;
            }

            REQUIRE(utils_list_to_array(&list, 0) == test_length);
            REQUIRE(utils_list_to_array(&list, copy) == test_length);

            for (c = 0; c < test_length; c++)
                REQUIRE(nodes[c] == copy[test_length - c - 1]);

            for (c = 0; c < test_length; c++) {
                utils_list_node_s *node = utils_list_pop_tail(&list);
                REQUIRE(nodes[c] == node);
                free_list_node(node);
            }
            REQUIRE(utils_list_pop_head(&list) == 0);
            REQUIRE(utils_list_pop_tail(&list) == 0);
            REQUIRE(list.length == 0);
        }

        REQUIRE(utils_list_pop_head(&list) == 0);
        REQUIRE(utils_list_pop_tail(&list) == 0);
        REQUIRE(list.length == 0);
    }
}
