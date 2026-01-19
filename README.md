# containerctl
KISS containers CLI/TUI manager, in Python

## Help
```
containerctl - KISS containers CLI/TUI manager
───────────────────────────────────────────────────────
- containerctl                       ==> launch TUI
- containerctl help                  ==> show this help
───────────────────────────────────────────────────────
- containerctl ps                    ==> list containers
- containerctl ps -q                 ==> list container IDs only
- containerctl ps -r                 ==> list running containers only
- containerctl images                ==> list images
- containerctl images -q             ==> list image IDs only
- containerctl volumes               ==> list volumes
- containerctl volumes -q            ==> list volume names only
- containerctl networks              ==> list networks
- containerctl networks -q           ==> list network IDs only
───────────────────────────────────────────────────────
- containerctl start <id>            ==> start container
- containerctl stop <id>             ==> stop container
- containerctl restart <id>          ==> restart container
───────────────────────────────────────────────────────
- containerctl logs <id>             ==> show container logs
- containerctl logs <id> -f          ==> follow container logs
- containerctl logs <id> -n 100      ==> show last 100 lines
───────────────────────────────────────────────────────
- containerctl exec <id> <cmd...>    ==> execute command in container
- containerctl shell <id>            ==> open shell in container
- containerctl shell <id> -s /bin/bash ==> open bash in container
───────────────────────────────────────────────────────
```
