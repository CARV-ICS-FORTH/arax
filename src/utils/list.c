#include "list.h"

utils_list_s* utils_list_init(void *mem)
{
	utils_list_s *list = mem;

	list->length = 0;
	utils_list_node_init(&(list->head));
	return list;
}

void utils_list_node_add(utils_list_node_s *head, utils_list_node_s *node)
{
	head->next->prev = node;
	node->next = head->next;
	node->prev = head;
	head->next = node;
}

void utils_list_add(utils_list_s *list, utils_list_node_s *node)
{
	utils_list_node_add(&(list->head),node);
	list->length++;
}

utils_list_node_s* utils_list_del(utils_list_s *list, utils_list_node_s *node)
{
	node->next->prev = node->prev;
	node->prev->next = node->next;
	list->length--;
	return node;
}

int utils_list_to_array(utils_list_s *list, utils_list_node_s **array)
{
	utils_list_node_s *itr;

	if (!array)
		return list->length;

	if (list->length) {
		utils_list_for_each(*list,itr)
		{
			*array = itr;
			array++;
		}
		return list->length;
	}
	array[0] = 0;
	return 0;
}

void utils_list_node_init(utils_list_node_s *node)
{
	node->next = node;
	node->prev = node;
}
