# containerctl
KISS containers CLI/TUI manager, in Python

## Help
```
containerctl - KISS containers CLI/TUI manager
──────────────────────────────────────────────
- containerctl                          ==> launch TUI
- containerctl help                     ==> show this help
- containerctl help config              ==> example project, service and shared env TOML
──────────────────────────────────────────────
- containerctl ps                       ==> list containers
- containerctl ps -q                    ==> list container IDs only
- containerctl ps -r                    ==> list running containers only
- containerctl ps -w                    ==> wide: never clip columns to the terminal width
- containerctl images                   ==> list images
- containerctl images -q                ==> list image IDs only
- containerctl volumes                  ==> list volumes
- containerctl volumes -q               ==> list volume names only
- containerctl networks                 ==> list networks
- containerctl networks -q              ==> list network IDs only
──────────────────────────────────────────────
- containerctl start <id>               ==> start container
- containerctl stop <id>                ==> stop container
- containerctl restart <id>             ==> restart container
──────────────────────────────────────────────
- containerctl logs <id>                ==> show container logs
- containerctl logs <id> -f             ==> follow container logs
- containerctl logs <id> -n 100         ==> show last 100 lines
──────────────────────────────────────────────
- containerctl exec <id> <cmd...>       ==> execute command in container
- containerctl shell <id>               ==> open shell in container
- containerctl shell <id> -s /bin/bash  ==> open bash in container
──────────────────────────────────────────────
- containerctl enter <project>          ==> enter project (spawn or attach)
- containerctl enter <proj> user=root   ==> enter project as another user
- containerctl enter <shortcut>         ==> enter project, landing in a [shortcuts] dir
- containerctl spawn <project>          ==> force-create project container
- containerctl destroy <project>        ==> stop and remove project container
- containerctl projects                 ==> list configured projects
- containerctl projects -w              ==> wide: full shortcut lists, never clip
- containerctl env                      ==> list the env injected at enter (names, never values)
- containerctl env <project>            ==> same, with the project's own [env] applied
──────────────────────────────────────────────
- containerctl check <project>          ==> is the workspace safe to destroy? (uncommitted/unpushed/extra)
- containerctl check --all              ==> check every configured project
- containerctl check-local              ==> run the same check from inside a container
──────────────────────────────────────────────
- containerctl service up <name>               ==> create/start a service
- containerctl service down <name>             ==> stop and remove a service
- containerctl service recreate <name>         ==> recreate a service
- containerctl service scripts <name>          ==> list pre-registered scripts
- containerctl service script <name> <script>  ==> run a pre-registered script
- containerctl services                        ==> list configured services
- containerctl services -w                     ==> wide: never clip columns
──────────────────────────────────────────────
Project configs: ~/.config/containerctl/projects/<name>.toml
Service configs: ~/.config/containerctl/services/<name>.toml
Shared env:      ~/.config/containerctl/config.toml (declares) + ~/.config/containerctl/secrets.toml (secret values, 0600)
```
