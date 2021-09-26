#ifndef UTILS_LIST_HEADER
#define UTILS_LIST_HEADER
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

typedef struct utils_list_node
{
    struct utils_list_node *next;  /**< Pointer to next list node */
    struct utils_list_node *prev;  /**< Pointer to prev list node */
    void *                  owner; /**< Pointer to owner */
} utils_list_node_s;
typedef struct
{
    utils_list_node_s head;   /**< Head node */
    size_t            length; /**< List length */
} utils_list_s;

/**
 * Initialize a utils_list_s instance in \c node.
 *
 * @param mem An allocated buffer of at least sizeof(utils_list_s) size.
 * @return Equal to \c mem if successful, NULL on failure.
 */
utils_list_s* utils_list_init(void *mem);

/**
 * Add \c node to \c list as the new head of the list.
 */
void utils_list_add(utils_list_s *list, utils_list_node_s *node);

/**
 * Delete \c node from list.
 * @return The deleted node, NULL on failure.
 */
utils_list_node_s* utils_list_del(utils_list_s *list, utils_list_node_s *node);

/**
 * Remove first node from \c list and return to caller.
 *
 * \note Not thread safe!
 *
 * @param list A valid utils_list_s instance.
 * @return The node that was first in \c list, NULL if list was empty
 */
utils_list_node_s* utils_list_pop_head(utils_list_s *list);

/**
 * Remove last node from \c list and return to caller.
 *
 * \note Not thread safe!
 *
 * @param list A valid utils_list_s instance.
 * @return The node that was last in \c list, NULL if list was empty
 */
utils_list_node_s* utils_list_pop_tail(utils_list_s *list);

/**
 * Convert list to array and return number of entries.
 *
 * If \c array is NULL just return the number of list node in \c list.
 * If \c array is not NULL, fill \c array with the node->owner values
 * of all nodes.
 *
 * @param list A valid utils_list_s instance.
 * @param array An array of pointers to all node->owner.
 * @return Number of elements in list.
 */
size_t utils_list_to_array(utils_list_s *list, void **array);

/**
 * Initialize a utils_list_node_s.
 *
 * @param node The utils_list_node_s to be initialized.
 * @param owner Pointer to the node 'usefull' data
 */
void utils_list_node_init(utils_list_node_s *node, void *owner);

/**
 * Return if \c node is part of some list.
 *
 * @param node The utils_list_node_s to be initialized.
 * @return 0 if not is not part of a list, non zero if it is part of a list.
 */
int utils_list_node_linked(utils_list_node_s *node);

/**
 * Iterate through a utils_list_s nodes.
 *
 * @param list Pointer to a valid utils_list_s instance.
 * @param itr A utils_list_node_s* variable.
 */
#define utils_list_for_each(list, itr) \
    for (itr = (list).head.next; itr != (void *) &list; itr = itr->next)

/**
 * Iterate through a utils_list_s nodes safely(can call utils_list_del).
 *
 * @param list Pointer to a valid utils_list_s instance.
 * @param itr A utils_list_node_s* variable pointing to the current element.
 * @param tmp A utils_list_node_s* variable pointing to the next element.
 */
#define utils_list_for_each_safe(list, itr, tmp) \
    for (itr = (list).head.next; (itr != (void *) &list) && (tmp = itr->next); itr = tmp)

/**
 * Iterate through a utils_list_s nodes.
 *
 * @param list Pointer to a valid utils_list_s instance.
 * @param itr A utils_list_node_s* variable.
 */
#define utils_list_for_each_reverse(list, itr) \
    for (itr = (list).head.prev; itr != (void *) &list; itr = itr->prev)

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef UTILS_LIST_HEADER */
