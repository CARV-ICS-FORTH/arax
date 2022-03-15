#ifndef VDF_MON_API_HEADER
#define VDF_MIN_API_HEADER
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <string.h>

#define TASK_TYPE 1
#define JOB_TYPE  2

typedef struct
{
    uint64_t timestamp; // Issue Time
    void *   accel;
    union {
        struct
        {
            uint64_t k_start; // just before host_code
            uint64_t k_end;   // just after host_code
        } task;
        struct
        {
            char     jobid[7];
            uint8_t  batch;
            uint64_t j_start;
            uint64_t j_end;
        } job;
    };
    uint8_t type;
} Mon;

static int sockfd = -1;

#define COLLECTOR "139.91.92.7"

static void monitorTask(void *accel, uint64_t issue_time, uint64_t k_start, uint64_t k_end)
{
    Mon msg;

    msg.type         = TASK_TYPE;
    msg.accel        = accel;
    msg.task.k_start = k_start;
    msg.task.k_end   = k_end;

    if (sockfd == -1) {
        static struct sockaddr_in serv_addr = { 0 };
        struct hostent *server;
        sockfd = socket(PF_INET, SOCK_STREAM, 0);
        serv_addr.sin_family      = PF_INET;
        serv_addr.sin_port        = htons(8889);
        serv_addr.sin_addr.s_addr = inet_addr(COLLECTOR);

        if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
            abort();
    }
    send(sockfd, &msg, sizeof(msg), 0);
}

static void monitorJob(void *accel, const char *jobid, uint8_t batch, uint64_t j_start, uint64_t j_end)
{
    Mon msg;

    msg.type  = JOB_TYPE;
    msg.accel = accel;
    strncpy(msg.job.jobid, jobid, 7);
    msg.job.j_start = j_start;
    msg.job.j_end   = j_end;

    if (sockfd == -1) {
        static struct sockaddr_in serv_addr = { 0 };
        struct hostent *server;
        sockfd = socket(PF_INET, SOCK_STREAM, 0);
        serv_addr.sin_family      = PF_INET;
        serv_addr.sin_port        = htons(8889);
        serv_addr.sin_addr.s_addr = inet_addr(COLLECTOR);
        if (connect(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0)
            abort();
    }
    send(sockfd, &msg, sizeof(msg), 0);
}

#endif // ifndef VDF_MON_API_HEADER
