#include "vine_assert.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <execinfo.h>
#include "system.h"

static char _stackFnStr[4096];

const char* formatStackLine(const char *bt_sym, int *cwidths)
{
    char *temp;
    char *pos = strdup(bt_sym);

    temp = pos;
    char *exe = pos;

    strtok(pos, "(");
    pos += strlen(pos) + 1;
    char *symbol = strtok(pos, ")");

    pos += strlen(pos) + 1;
    char *addr = strtok(pos, "[");

    pos += strlen(addr) + 1;
    strtok(pos, "]");
    addr = pos;

    if (cwidths[0] < strlen(exe)) cwidths[0] = strlen(exe);
    if (cwidths[1] < strlen(addr)) cwidths[1] = strlen(addr);
    if (cwidths[2] < strlen(symbol)) cwidths[2] = strlen(symbol);

    sprintf(_stackFnStr, "%*s %*s %s", cwidths[0], exe, cwidths[1], addr, symbol);

    free(temp);

    return _stackFnStr;
}

#define FMT "%*s%*s"
#define MID(LEN, STR) (int) (LEN / 2 + strlen(STR)), STR, (int) (LEN / 2 - strlen(STR)), " "

#define VINE_FILE_PREFIX_LEN (strlen(__FILE__) - 23)
// GCOV_EXCL_START
void _vine_assert(int value, const char *expr, const char *file, int line)
{
    if (!value)
        return;

    void *bt[128];
    char **bt_syms;
    int bt_size = 128;
    int bt_indx;
    int cwidths[3] = { 6, 6, 8 };

    bt_size = backtrace(bt, bt_size);
    bt_syms = backtrace_symbols(bt, bt_size);

    // Do it once to get column widths
    for (bt_indx = bt_size - 1; bt_indx; bt_indx--)
        formatStackLine(bt_syms[bt_indx], cwidths);

    fprintf(stderr, "\n\n" FMT FMT FMT, MID(cwidths[0], "Binary"), MID(cwidths[1], "Location"), MID(cwidths[2],
      "Symbol"));
    for (bt_indx = bt_size - 1; bt_indx; bt_indx--) {
        fprintf(stderr, "\n%s", formatStackLine(bt_syms[bt_indx], cwidths) );
    }
    fprintf(stderr, " <<\n\nvine_assert(%s) @ %s:%d\n\n", expr, file + VINE_FILE_PREFIX_LEN, line);
    abort();
}

// GCOV_EXCL_STOP
