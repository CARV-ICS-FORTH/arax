#include "system.h"
#include <sys/stat.h>
#include <string.h>
#include <execinfo.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <pwd.h>
#include <fcntl.h>
#include <sys/mman.h>

size_t system_total_memory()
{
    size_t pages     = sysconf(_SC_PHYS_PAGES);
    size_t page_size = sysconf(_SC_PAGE_SIZE);

    return pages * page_size;
}

char* system_home_path()
{
    uid_t uid         = getuid();
    struct passwd *pw = getpwuid(uid);

    if (!pw)       // GCOV_EXCL_LINE
        return 0;  // GCOV_EXCL_LINE

    return pw->pw_dir;
}

const char* system_env_var(const char *var)
{
    return getenv(var);
}

int system_mmap(void **base, int *fd, const char *file, size_t shm_size, size_t shm_off, int truncate)
{
    if (*fd <= 0) {
        if (file[0] == '/')
            *fd = open(file, O_CREAT | O_RDWR, S_IRWXU);
        else
            *fd = shm_open(file, O_CREAT | O_RDWR, S_IRWXU);
    }

    if (fd < 0)
        return 1;

    if (truncate && (system_file_size(file) != shm_size)) { /* If not the correct size */
        if (ftruncate(*fd, shm_size) )
            return 1;
    }

    #ifdef MMAP_POPULATE
    *base = mmap(*base, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, *fd, shm_off);
    #else
    *base = mmap(*base, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, shm_off);
    #endif

    if (MAP_FAILED == *base) {
        *base = 0;
        return 1;
    }

    return 0;
}

off_t system_file_size(const char *file)
{
    struct stat stats = { 0 };

    if (stat(file, &stats))
        return 0;

    return stats.st_size;
}

const char* system_exec_name()
{
    static char exec_name[1024];
    const char *proc_exe = "/proc/self/exe";
    size_t size = readlink(proc_exe, exec_name, 1023);

    if (size == -1)
        snprintf(exec_name, 1023, "%s: Could not readlink!\n", proc_exe);
    else
        exec_name[size] = 0;
    return exec_name;
}

int system_process_id()
{
    return getpid();
}

int system_thread_id()
{
    return syscall(SYS_gettid);
}

char* formatStackLine(const char *bt_sym, int *cwidths, char *dest)
{
    char *temp;
    char *pos         = strdup(bt_sym);
    size_t bt_sym_len = strlen(bt_sym);

    for (temp = pos; *temp != 0; temp++) {
        switch (*temp) {
            case '(':
            case ')':
            case '[':
            case ']':
                *temp = ' ';
        }
    }

    char *exe    = malloc(bt_sym_len);
    char *symbol = malloc(bt_sym_len);
    char *addr   = malloc(bt_sym_len);

    sprintf(symbol, "NO_SYMB");
    sprintf(addr, "NO_ADDR");

    sscanf(pos, "%s %s %s", exe, symbol, addr);

    if (strcmp(addr, "NO_SYMB") == 0) {
        void *temp = addr;
        addr   = symbol;
        symbol = temp;
        sprintf(symbol, "NO_SYMB");
    }

    if (cwidths[0] < strlen(exe)) cwidths[0] = strlen(exe);
    if (cwidths[1] < strlen(addr)) cwidths[1] = strlen(addr);
    if (cwidths[2] < strlen(symbol)) cwidths[2] = strlen(symbol);

    if (dest)
        dest += sprintf(dest, "%*s %*s %s", cwidths[0], exe, cwidths[1], addr, symbol);


    free(addr);
    free(symbol);
    free(exe);
    free(pos);

    return dest;
} /* formatStackLine */

#define FMT "%*s%*s"
#define MID(LEN, STR) (int) (LEN / 2 + strlen(STR)), STR, (int) (LEN / 2 - strlen(STR)), " "

static char __backtraceStr[32768];

const char* system_backtrace(unsigned int skip)
{
    void *bt[128];
    char **bt_syms;
    int bt_size = 128;
    int bt_indx;
    int cwidths[3] = { 6, 6, 8 };
    char *dest     = __backtraceStr;

    bt_size = backtrace(bt, bt_size);

    if (bt_size > 0) {
        sprintf(__backtraceStr, "Could not generate backtrace\n");
        return __backtraceStr;
    }
    bt_syms = backtrace_symbols(bt, bt_size);

    // Do it once to get column widths
    for (bt_indx = bt_size - 1; bt_indx != skip; bt_indx--)
        formatStackLine(bt_syms[bt_indx], cwidths, 0);

    dest +=
      sprintf(dest, "\n\n" FMT FMT FMT, MID(cwidths[0], "Binary"), MID(cwidths[1], "Location"), MID(cwidths[2],
        "Symbol"));
    for (bt_indx = bt_size - 1; bt_indx != skip; bt_indx--) {
        *dest = '\n';
        dest++;
        dest = formatStackLine(bt_syms[bt_indx], cwidths, dest);
    }

    free(bt_syms);

    return __backtraceStr;
} /* system_backtrace */
