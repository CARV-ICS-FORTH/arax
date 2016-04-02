#include "list.h"

utils_list_s* utils_list_init(void *mem)
{
	utils_list_s *list = mem;

	list->length = 0;
	list->next   = 0;

	return list;
}

void utils_list_add(utils_list_s *list, utils_list_node_s *node)
{
	node->next = list->next;
	list->next = node;
	list->length++;
}

int utils_list_to_array(utils_list_s *list, void **array)
{
	utils_list_node_s *node = list->next;

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

void utils_list_node_init(utils_list_node_s *node)
{
	node->next = 0;
}
