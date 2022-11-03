/*
 * sysstat: System performance tools for Linux
 * (C) 1999-2017 by Sebastien Godard (sysstat <at> orange.fr)
 */

#ifndef _COMMON_H
#define _COMMON_H

/* Maximum length of sensors device name */
#define MAX_SENSORS_DEV_LEN 20

#include <limits.h>
#include <sched.h> /* For __CPU_SETSIZE */
#include <stdlib.h>
#include <time.h>

#ifdef HAVE_SYS_SYSMACROS_H
/* Needed on some non-glibc environments */
#include <sys/sysmacros.h>
#endif

#include "rd_stats.h"

/*
 ***************************************************************************
 * Various keywords and constants
 ***************************************************************************
 */

#define FALSE        0
#define TRUE         1

#define PLAIN_OUTPUT 0

#define DISP_HDR     1

/* Index in units array (see common.c) */
#define NO_UNIT       -1
#define UNIT_SECTOR   0
#define UNIT_BYTE     1
#define UNIT_KILOBYTE 2

#define NR_UNITS      8

/* Timestamp buffer length */
#define TIMESTAMP_LEN 64

/* Number of seconds per day */
#define SEC_PER_DAY 3600 * 24

/* Maximum number of CPUs */
#if defined(__CPU_SETSIZE) && __CPU_SETSIZE > 8192
#define NR_CPUS __CPU_SETSIZE
#else
#define NR_CPUS 8192
#endif

/* Maximum number of interrupts */
#define NR_IRQS 1024

/* Size of /proc/interrupts line, CPU data excluded */
#define INTERRUPTS_LINE 128

/* Keywords */
#define K_ISO      "ISO"
#define K_ALL      "ALL"
#define K_LOWERALL "all"
#define K_UTC      "UTC"
#define K_JSON     "JSON"

/* Files */
#define STAT                   "/proc/stat"
#define UPTIME                 "/proc/uptime"
#define DISKSTATS              "/proc/diskstats"
#define INTERRUPTS             "/proc/interrupts"
#define MEMINFO                "/proc/meminfo"
#define SYSFS_BLOCK            "/sys/block"
#define SYSFS_DEV_BLOCK        "/sys/dev/block"
#define SYSFS_DEVCPU           "/sys/devices/system/cpu"
#define SYSFS_TIME_IN_STATE    "cpufreq/stats/time_in_state"
#define S_STAT                 "stat"
#define DEVMAP_DIR             "/dev/mapper"
#define DEVICES                "/proc/devices"
#define SYSFS_USBDEV           "/sys/bus/usb/devices"
#define DEV_DISK_BY            "/dev/disk/by"
#define SYSFS_IDVENDOR         "idVendor"
#define SYSFS_IDPRODUCT        "idProduct"
#define SYSFS_BMAXPOWER        "bMaxPower"
#define SYSFS_MANUFACTURER     "manufacturer"
#define SYSFS_PRODUCT          "product"
#define SYSFS_FCHOST           "/sys/class/fc_host"

#define MAX_FILE_LEN           256
#define MAX_PF_NAME            1024
#define MAX_NAME_LEN           128

#define IGNORE_VIRTUAL_DEVICES FALSE
#define ACCEPT_VIRTUAL_DEVICES TRUE

/* Environment variables */
#define ENV_TIME_FMT       "S_TIME_FORMAT"
#define ENV_TIME_DEFTM     "S_TIME_DEF_TIME"
#define ENV_COLORS         "S_COLORS"
#define ENV_COLORS_SGR     "S_COLORS_SGR"

#define C_NEVER            "never"
#define C_ALWAYS           "always"

#define DIGITS             "0123456789"
#define PERCENT_LIMIT_HIGH 75.0
#define PERCENT_LIMIT_LOW  50.0
#define MAX_SGR_LEN        16

#define C_LIGHT_RED        "\e[31;22m"
#define C_BOLD_RED         "\e[31;1m"
#define C_LIGHT_GREEN      "\e[32;22m"
#define C_LIGHT_YELLOW     "\e[33;22m"
#define C_BOLD_MAGENTA     "\e[35;1m"
#define C_BOLD_BLUE        "\e[34;1m"
#define C_LIGHT_BLUE       "\e[34;22m"
#define C_NORMAL           "\e[0m"

#define SP_VALUE(m, n, p) (((double) ((n) - (m))) / (p) * 100)
#define HZ hz

extern unsigned int hz;
extern "C" {
extern int cpu_nr;
}
double ll_sp_value(unsigned long long, unsigned long long, unsigned long long);

void cprintf_pc(int, int, int, int, ...);
int is_iso_time_fmt(void);
void get_HZ(void);
unsigned long long get_per_cpu_interval(struct stats_cpu *, struct stats_cpu *);
time_t get_localtime(struct tm *, int);

#endif // ifndef _COMMON_H
