#ifndef ATRING_HEADER
	#define ATRING_HEADER

	typedef void atring;

	/**
	 * Initialize a atring at the memory pointed by buff.
	 *
	 * @buff Allocated buffer.
	 * @bytes Size of provided buffer to be used.
	 * @return atring instance.NULL on failure.
	 */
	atring * atring_init(void * buff,int bytes);

	/**
	 * Calculate byte allocation required for an atring with specified slots.
	 *
	 * @slots Number of slots in the atring.
	 * @return Size of required buffer size able to fit the slots.
	 */
	int atring_calc_bytes(int slots);

	/**
	 * Return number of unused slots in the atring.
	 *
	 * @slots Valid atring instance pointer.
	 * @return Number of free slots in atring.
	 */
	int atring_free_slots(atring * ar);
	/**
	 * Return number of used slots in the atring.
	 *
	 * @slots Valid atring instance pointer.
	 * @return Number of used slots in atring.
	 */
	int atring_used_slots(atring * ar);
	/**
	 * Add data to an atring
	 *
	 * @slots Valid atring instance pointer.
	 * @data Non NULL pointer to data.
	 * @return Equal to data, NULL on failure.
	 */
	void * atring_push(atring* ar,void * data);
	/**
	 * Pop data from atring.
	 *
	 * @ar Valid atring instance pointer.
	 * @return Data pointer, NULL on failure.
	 */
	void * atring_pop(atring* ar);
#endif
