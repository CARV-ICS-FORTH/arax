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

utils_list_s* utils_list_init(void *mem);

void utils_list_add(utils_list_s *list, utils_list_node_s *node);

int utils_list_to_array(utils_list_s *list, void **array);

void utils_list_node_init(utils_list_node_s *node);

/* FIXME: This is kind of ugly and dangerous */
#define utils_list_for_each(list, itr) \
	for (itr = (list).next; itr; itr = itr->next)

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef UTILS_LIST_HEADER */
