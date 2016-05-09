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

utils_list_node_s* utils_list_del(utils_list_s *list, utils_list_node_s *node)
{
	utils_list_node_s *prev = (utils_list_node_s*)&(list->next);

	if ( !(prev->next) )
		return 0;
	while (prev && prev->next != node) // Might want to change to d linked
		                           // list
		prev = prev->next;
	if (!prev)
		return 0;
	prev->next = node->next; // Delete node
	list->length--;
	return node;
}

int utils_list_to_array(utils_list_s *list, utils_list_node_s **array)
{
	utils_list_node_s *node = list->next;

	if (!array)
		return list->length;

	if (node) {
		do {
			array[0] = node;
			array++;
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
