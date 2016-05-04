#ifndef UTILS_LIST_HEADER
#define UTILS_LIST_HEADER
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

typedef struct {
	uint64_t length; /**< List length(intentionally 8 bytes) */
	void     *next; /**< Pointer to next list node */
} utils_list_s;

typedef struct {
	void *next; /**< Pointer to next list node */
} utils_list_node_s;

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
utils_list_node_s * utils_list_del(utils_list_s *list, utils_list_node_s *node);

/**
 * Convert list to array and return number of list nodes.
 *
 * If \c array is NULL just return the number of list node in \c list.
 * If \c array is not NULL, fill \c array with pointers to the
 * utils_list_node_s of \c list
 *
 * @param list A valid utils_list_s instance.
 * @param array An array of pointers to the \c list list_node.
 */
int utils_list_to_array(utils_list_s *list, utils_list_node_s **array);

/**
 * Initialize a utils_list_node_s.
 *
 * @param node The utils_list_node_s to be initialized.
 */
void utils_list_node_init(utils_list_node_s *node);

/**
 * Iterate through a utils_list_s nodes.
 *
 * @param list Pointer to a valid utils_list_s instance.
 * @param itr A utils_list_node_s* variable.
 */
#define utils_list_for_each(list, itr) \
	for (itr = (list).next; itr; itr = itr->next)

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef UTILS_LIST_HEADER */
