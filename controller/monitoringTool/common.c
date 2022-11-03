/*
 * sar, sadc, sadf, mpstat and iostat common routines.
 * (C) 1999-2017 by Sebastien GODARD (sysstat <at> orange.fr)
 *
 ***************************************************************************
 * This program is free software; you can redistribute it and/or modify it *
 * under the terms of the GNU General Public License as published  by  the *
 * Free Software Foundation; either version 2 of the License, or (at  your *
 * option) any later version.                                              *
 *                                                                         *
 * This program is distributed in the hope that it  will  be  useful,  but *
 * WITHOUT ANY WARRANTY; without the implied warranty  of  MERCHANTABILITY *
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License *
 * for more details.                                                       *
 *                                                                         *
 * You should have received a copy of the GNU General Public License along *
 * with this program; if not, write to the Free Software Foundation, Inc., *
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1335 USA              *
 ***************************************************************************
 */

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <inttypes.h>
#include <libgen.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h> /* For STDOUT_FILENO, among others */

//#include "version.h"
#include "common.h"
//#include "ioconf.h"
#include "rd_stats.h"

#ifdef USE_NLS
#include <libintl.h>
#include <locale.h>
#define _(string) gettext(string)
#else
#define _(string) (string)
#endif

int cpu_nr;
/* Units (sectors, Bytes, kilobytes, etc.) */
char units[] = {'s', 'B', 'k', 'M', 'G', 'T', 'P', '?'};

/* Number of ticks per second */
unsigned int hz;
/* Number of bit shifts to convert pages to kB */
unsigned int kb_shift;

/* Colors strings */
char sc_percent_high[MAX_SGR_LEN] = C_BOLD_RED;
char sc_percent_low[MAX_SGR_LEN] = C_BOLD_MAGENTA;
char sc_zero_int_stat[MAX_SGR_LEN] = C_LIGHT_BLUE;
char sc_int_stat[MAX_SGR_LEN] = C_BOLD_BLUE;
char sc_item_name[MAX_SGR_LEN] = C_LIGHT_GREEN;
char sc_sa_restart[MAX_SGR_LEN] = C_LIGHT_RED;
char sc_sa_comment[MAX_SGR_LEN] = C_LIGHT_YELLOW;
char sc_normal[MAX_SGR_LEN] = C_NORMAL;

/*
 ***************************************************************************
 * Print "percent" statistics values using colors.
 *
 * IN:
 * @human	Set to > 0 if a percent sign (%) shall be displayed after
 *		the value.
 * @num		Number of values to print.
 * @wi		Output width.
 * @wd		Number of decimal places.
 ***************************************************************************
 */
void cprintf_pc(int human, int num, int wi, int wd, ...) {
  printf("cprintf\n");
  int i;
  double val, lim = 0.005;
  char u = '\0';
  va_list args;

  /*
   * If a percent sign is to be displayed, then there will be only one decimal
   * place. In this case, a value smaller than 0.05 shall be considered as 0.
   */
  if (human > 0) {
    lim = 0.05;
    u = '%';
    if (wi < 4) {
      /* E.g., 100% */
      wi = 4;
    }
    /* Keep one place for the percent sign */
    wi -= 1;
    if (wd > 0) {
      wd -= 1;
    }
  }

  va_start(args, wd);

  for (i = 0; i < num; i++) {
    val = va_arg(args, double);
    if (val >= PERCENT_LIMIT_HIGH) {
      printf("%s", sc_percent_high);
    } else if (val >= PERCENT_LIMIT_LOW) {
      printf("%s", sc_percent_low);
    } else if (val < lim) {
      printf("%s", sc_zero_int_stat);
    } else {
      printf("%s", sc_int_stat);
    }
    printf(" %*.*f", wi, wd, val);
    printf("%s", sc_normal);
    printf("%c", u);
  }

  va_end(args);
}
/*
 ***************************************************************************
 * Workaround for CPU counters read from /proc/stat: Dyn-tick kernels
 * have a race issue that can make those counters go backward.
 ***************************************************************************
 */
double ll_sp_value(unsigned long long value1, unsigned long long value2,
                   unsigned long long itv) {
  if (value2 < value1)
    return (double)0;
  else
    return SP_VALUE(value1, value2, itv);
}

/*
 ***************************************************************************
 * Returns whether S_TIME_FORMAT is set to ISO.
 *
 * RETURNS:
 * TRUE if S_TIME_FORMAT is set to ISO, or FALSE otherwise.
 ***************************************************************************
 */
int is_iso_time_fmt(void) {
  static int is_iso = -1;
  char *e;

  if (is_iso < 0) {
    is_iso = (((e = getenv(ENV_TIME_FMT)) != NULL) && !strcmp(e, K_ISO));
  }
  return is_iso;
}

/*
 ***************************************************************************
 * Compute time interval.
 *
 * IN:
 * @prev_uptime	Previous uptime value in jiffies.
 * @curr_uptime	Current uptime value in jiffies.
 *
 * RETURNS:
 * Interval of time in jiffies.
 ***************************************************************************
 */
unsigned long long get_interval(unsigned long long prev_uptime,
                                unsigned long long curr_uptime) {
  unsigned long long itv;

  /* prev_time=0 when displaying stats since system startup */
  itv = curr_uptime - prev_uptime;
  if (!itv) { /* Paranoia checking */
    itv = 1;
  }
  return itv;
}

/*
 *  ***************************************************************************
 *   * Get number of clock ticks per second.
 *    ***************************************************************************
 *     */
void get_HZ() {
  long ticks;

  if ((ticks = sysconf(_SC_CLK_TCK)) == -1) {
    perror("sysconf");
  }
  hz = (unsigned int)ticks;
}

/*
 ***************************************************************************
 * Since ticks may vary slightly from CPU to CPU, we'll want
 * to recalculate itv based on this CPU's tick count, rather
 * than that reported by the "cpu" line. Otherwise we
 * occasionally end up with slightly skewed figures, with
 * the skew being greater as the time interval grows shorter.
 *
 * IN:
 * @scc	Current sample statistics for current CPU.
 * @scp	Previous sample statistics for current CPU.
 *
 * RETURNS:
 * Interval of time based on current CPU.
 ***************************************************************************
 */
unsigned long long get_per_cpu_interval(struct stats_cpu *scc,
                                        struct stats_cpu *scp) {
  unsigned long long ishift = 0LL;

  if ((scc->cpu_user - scc->cpu_guest) < (scp->cpu_user - scp->cpu_guest)) {
    /*
     * Sometimes the nr of jiffies spent in guest mode given by the guest
     * counter in /proc/stat is slightly higher than that included in
     * the user counter. Update the interval value accordingly.
     */
    ishift +=
        (scp->cpu_user - scp->cpu_guest) - (scc->cpu_user - scc->cpu_guest);
  }
  if ((scc->cpu_nice - scc->cpu_guest_nice) <
      (scp->cpu_nice - scp->cpu_guest_nice)) {
    /*
     * Idem for nr of jiffies spent in guest_nice mode.
     */
    ishift += (scp->cpu_nice - scp->cpu_guest_nice) -
              (scc->cpu_nice - scc->cpu_guest_nice);
  }

  /*
   * Don't take cpu_guest and cpu_guest_nice into account
   * because cpu_user and cpu_nice already include them.
   */
  return (
      (scc->cpu_user + scc->cpu_nice + scc->cpu_sys + scc->cpu_iowait +
       scc->cpu_idle + scc->cpu_steal + scc->cpu_hardirq + scc->cpu_softirq) -
      (scp->cpu_user + scp->cpu_nice + scp->cpu_sys + scp->cpu_iowait +
       scp->cpu_idle + scp->cpu_steal + scp->cpu_hardirq + scp->cpu_softirq) +
      ishift);
}
/*
 ***************************************************************************
 * Get local date and time.
 *
 * IN:
 * @d_off	Day offset (number of days to go back in the past).
 *
 * OUT:
 * @rectime	Current local date and time.
 *
 * RETURNS:
 * Value of time in seconds since the Epoch.
 ***************************************************************************
 */
time_t get_localtime(struct tm *rectime, int d_off) {
  time_t timer;
  struct tm *ltm;

  time(&timer);
  timer -= SEC_PER_DAY * d_off;
  ltm = localtime(&timer);

  if (ltm) {
    *rectime = *ltm;
  }
  return timer;
}
