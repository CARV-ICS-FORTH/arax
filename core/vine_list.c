#include "vine_list.h"

vine_list_s * vine_list_init(void * mem)
{
	vine_list_s * list = mem;
	list->length = 0;
	list->next = 0;
}

void vine_list_add(vine_list_s * list,vine_list_node_s * node)
{
	node->next = list->next;
	list->next = node;
	list->length++;
}

int vine_list_to_array(vine_list_s * list,void ** array)
{
	vine_list_node_s * node = list->next;

	if(!array)
		return list->length;

	if(node)
	{
		do
		{
			array[0] = node;
			(*array)++;
			node = node->next;
		}while(node);
		return list->length;
	}
	array[0] = 0;
	return 0;
}

void vine_list_node_init(vine_list_node_s * node)
{
	node->next = 0;
}
