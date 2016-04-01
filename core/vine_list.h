#ifndef VINE_LIST_HEADER
#define VINE_LIST_HEADER
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct
{
	uint64_t length;	/**< List length(intentionally 8 bytes) */
	void * next;		/**< Pointer to next list node */
}vine_list_s;

typedef struct
{
	void * next;		/**< Pointer to next list node */
}vine_list_node_s;

vine_list_s * vine_list_init(void * mem);

void vine_list_add(vine_list_s * list,vine_list_node_s * node);

int vine_list_to_array(vine_list_s * list,void ** array);

void vine_list_node_init(vine_list_node_s * node);

#define vine_list_for_each(list,itr) \
	for(itr = (list).next ; itr ; itr = itr->next)


#ifdef __cplusplus
}
#endif

#endif
