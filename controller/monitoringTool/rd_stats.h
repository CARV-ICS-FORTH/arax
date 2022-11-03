/*
 * rd_stats.h: Include file used to read system statistics
 * (C) 1999-2017 by Sebastien Godard (sysstat <at> orange.fr)
 */

#ifndef _RD_STATS_H
#define _RD_STATS_H

/*
 ***************************************************************************
 * Miscellaneous constants
 ***************************************************************************
 */

/* Get IFNAMSIZ */
#include <net/if.h>
#ifndef IFNAMSIZ
#define IFNAMSIZ 16
#endif

/* Maximum length of network interface name */
#define MAX_IFACE_LEN IFNAMSIZ
/* Maximum length of USB manufacturer string */
#define MAX_MANUF_LEN 24
/* Maximum length of USB product string */
#define MAX_PROD_LEN 48
/* Maximum length of filesystem name */
#define MAX_FS_LEN 128
/* Maximum length of FC host name */
#define MAX_FCH_LEN 16

#define CNT_PART 1
#define CNT_ALL_DEV 0
#define CNT_USED_DEV 1

#define K_DUPLEX_HALF "half"
#define K_DUPLEX_FULL "full"

#define C_DUPLEX_HALF 1
#define C_DUPLEX_FULL 2

/*
 ***************************************************************************
 * System files containing statistics
 ***************************************************************************
 */

/* Files */
#define PROC "/proc"
#define SERIAL "/proc/tty/driver/serial"
#define FDENTRY_STATE "/proc/sys/fs/dentry-state"
#define FFILE_NR "/proc/sys/fs/file-nr"
#define FINODE_STATE "/proc/sys/fs/inode-state"
#define PTY_NR "/proc/sys/kernel/pty/nr"
#define NET_DEV "/proc/net/dev"
#define NET_SOCKSTAT "/proc/net/sockstat"
#define NET_SOCKSTAT6 "/proc/net/sockstat6"
#define NET_RPC_NFS "/proc/net/rpc/nfs"
#define NET_RPC_NFSD "/proc/net/rpc/nfsd"
#define NET_SOFTNET "/proc/net/softnet_stat"
#define LOADAVG "/proc/loadavg"
#define VMSTAT "/proc/vmstat"
#define NET_SNMP "/proc/net/snmp"
#define NET_SNMP6 "/proc/net/snmp6"
#define CPUINFO "/proc/cpuinfo"
#define MTAB "/etc/mtab"
#define IF_DUPLEX "/sys/class/net/%s/duplex"
#define IF_SPEED "/sys/class/net/%s/speed"
#define FC_RX_FRAMES "%s/%s/statistics/rx_frames"
#define FC_TX_FRAMES "%s/%s/statistics/tx_frames"
#define FC_RX_WORDS "%s/%s/statistics/rx_words"
#define FC_TX_WORDS "%s/%s/statistics/tx_words"

/*
 ***************************************************************************
 * Definitions of structures for system statistics
 ***************************************************************************
 */

#define ULL_ALIGNMENT_WIDTH 16
#define UL_ALIGNMENT_WIDTH 8
#define U_ALIGNMENT_WIDTH 4

/*
 * Structure for CPU statistics.
 * In activity buffer: First structure is for global CPU utilisation ("all").
 * Following structures are for each individual CPU (0, 1, etc.)
 */
struct stats_cpu {
  unsigned long long cpu_user __attribute__((aligned(16)));
  unsigned long long cpu_nice __attribute__((aligned(16)));
  unsigned long long cpu_sys __attribute__((aligned(16)));
  unsigned long long cpu_idle __attribute__((aligned(16)));
  unsigned long long cpu_iowait __attribute__((aligned(16)));
  unsigned long long cpu_steal __attribute__((aligned(16)));
  unsigned long long cpu_hardirq __attribute__((aligned(16)));
  unsigned long long cpu_softirq __attribute__((aligned(16)));
  unsigned long long cpu_guest __attribute__((aligned(16)));
  unsigned long long cpu_guest_nice __attribute__((aligned(16)));
};

void read_uptime(unsigned long long *);
void read_stat_cpu(struct stats_cpu *, int, unsigned long long *,
                   unsigned long long *);
unsigned long long get_interval(unsigned long long, unsigned long long);
unsigned long long get_per_cpu_interval(struct stats_cpu *, struct stats_cpu *);
#define STATS_CPU_SIZE (sizeof(struct stats_cpu))
#endif
