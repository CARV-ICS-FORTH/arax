#ifndef STRUCTS_LIST_HEADER
#define STRUCTS_LIST_HEADER
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

typedef struct {
	uint64_t length; /**< List length(intentionally 8 bytes) */
	void     *next; /**< Pointer to next list node */
} structs_list_s;
typedef struct {
	void *next; /**< Pointer to next list node */
} structs_list_node_s;

structs_list_s* structs_list_init(void *mem);

void structs_list_add(structs_list_s *list, structs_list_node_s *node);

int structs_list_to_array(structs_list_s *list, void **array);

void structs_list_node_init(structs_list_node_s *node);

/* FIXME: This is kind of ugly and dangerous */
#define structs_list_for_each(list, itr) \
	for (itr = (list).next; itr; itr = itr->next)

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif /* ifndef STRUCTS_LIST_HEADER */
