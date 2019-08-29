#ifndef UTILS_VINE_ASSERT_HEADER
#define UTILS_VINE_ASSERT_HEADER

void _vine_assert(int value,const char * expr,const char * file,int line);

#define vine_assert(EXPR) \
	_vine_assert(!(EXPR),#EXPR,__FILE__,__LINE__)


#endif
