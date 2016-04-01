#include "list.h"

structs_list_s* structs_list_init(void *mem)
{
	structs_list_s *list = mem;

	list->length = 0;
	list->next   = 0;

	return list;
}

void structs_list_add(structs_list_s *list, structs_list_node_s *node)
{
	node->next = list->next;
	list->next = node;
	list->length++;
}

int structs_list_to_array(structs_list_s *list, void **array)
{
	structs_list_node_s *node = list->next;

	if (!array)
		return list->length;

	if (node) {
		do {
			array[0] = node;
			(*array)++;
			node = node->next;
		} while (node);
		return list->length;
	}
	array[0] = 0;
	return 0;
}

void structs_list_node_init(structs_list_node_s *node)
{
	node->next = 0;
}
