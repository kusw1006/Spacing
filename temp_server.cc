#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/un.h>
#include <pthread.h>
#include <sys/types.h> 
#include <arpa/inet.h>
#include <sys/socket.h>

#define BUF_SIZE    100
#define NAME_SIZE   20

void *send_msg(void *arg);
void *recv_msg(void *arg);

char name[NAME_SIZE] = "[DEFAULT]";
char msg[BUF_SIZE];

int main(int argc, char **argv) {
    int sock;
    struct sockaddr_in serv_addr;
    pthread_t snd_thread, rcv_thread;

    if(argc != 4) {
        printf("usage : %s <IP> <PORT> <NAME>\n", argv[0]);
        return -1;
    }    

    sprintf(name, "[%s]", argv[3]);
    sock = socket(AF_INET, SOCK_STREAM, 0);

    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr(argv[1]);
    serv_addr.sin_port = htons(atoi(argv[2]));

    if(connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("connect");
        return -1;
    }
    
    pthread_create(&snd_thread, NULL, send_msg, (void *)&sock);
    pthread_create(&rcv_thread, NULL, recv_msg, (void *)&sock);
    pthread_join(snd_thread, NULL);
    pthread_join(rcv_thread, NULL);
    close(sock);
    return 0;
}

void *send_msg(void *arg)
{
    int sock = *((int *)arg);
    char name_msg[NAME_SIZE+BUF_SIZE];
    while(1) {
        fgets(msg, BUF_SIZE, stdin);
        if(!strcmp(msg, "q\n") || !strcmp(msg, "Q\n")) {
            close(sock);
            // exit(0)가 아닌 return NULL로 하면 아직 recv_msg 스레드가 살아있어서 바로 종료되지 않는다
            exit(0);
        } 
        sprintf(name_msg, "%s %s", name, msg);
        write(sock, name_msg, strlen(name_msg));
    }
    return NULL;
}

void *recv_msg(void *arg)
{
    int sock = *((int *)arg);
    char name_msg[NAME_SIZE+BUF_SIZE];
    int str_len;
    while(1) {
        str_len = read(sock, name_msg, sizeof(name_msg) - 1);
        if(str_len < 0) {
            return NULL;
        }
        name_msg[str_len] = 0;
        fputs(name_msg, stdout);
    }

    return NULL;
}