# containerctl
KISS containers CLI/TUI manager, in Python

## Help
`containerctl help full` (this), `containerctl help` (the commands alone), `containerctl help config` (every TOML field).
```
containerctl - KISS containers CLI/TUI manager
──────────────────────────────────────────────
Why: I work in throwaway containers and keep a few services up. I want that in one file I can read,
stdlib only, on top of the docker CLI I already know. No daemon of its own, no state of its own:
what exists is what `docker ps` says, what should exist is a few TOML files. Not compose, not k8s.
──────────────────────────────────────────────
Concepts
- project  ==> a throwaway dev container (cctl-prj-<name>, own network cctl-net-prj-<name>) from a TOML:
               image, ssh key, repos cloned in ~/workspace, env files. `enter` spawns it when
               missing, then drops you in a shell. Nothing is mounted from the host: what is not pushed
               dies with the container. That is the point, and why `check` exists. It is made as
               cctl-spawning-<name> and renamed once all of that is in: a spawn that died halfway is
               not a project container: the next `enter` says how old it is, asks, and starts over.
- check    ==> is it safe to destroy? Everything in the workspace must be a configured repo, and every
               repo boring: nothing uncommitted or untracked, no stash, no detached HEAD, no linked
               worktree, every local branch on its upstream. Anything else is DIRTY, what could not be
               read included. Submodules, nested ones too, are repos of their own and checked as such.
               No network: it trusts the remote refs as last fetched. `destroy` runs it first, then asks.
- shortcut ==> a name for a dir of a project: `enter <shortcut>` lands there. Global: keep them unique.
- env      ==> of a project: [env] plain values, [secrets] names of keys in secrets.toml. Both reach the
               `enter` shell only: never argv, never the container's config, never a project file.
- service  ==> a long-lived container (cctl-svc-<name>): `docker run` written as TOML, plus scripts to
               exec in it. `up` creates or starts (and connects new networks), it never changes a
               container: any other TOML edit needs `recreate`. Its `env` is plain `docker run -e`:
               anyone who can `docker inspect` reads it.
- image    ==> a recipe: a Dockerfile (+ context dir) beside a TOML. Each build is its own tag,
               <repo>:cctl-local-NNNNNN, and the recipe tag is re-pointed at it: the previous build is
               still there to go back to. `spawn` / `up` build a missing image, never rebuild one.
               `image check` says when the build is stale, and diffs the Dockerfile against upstream.
               A project image must have: the project's user and its home (`dev` and /home/dev,
               unless its TOML says another `user` / `home`), bash, git (+ ssh for git@ repos),
               coreutils. python3 too, for `check-local` to run inside. Its USER does not matter: the
               clones, the checks and the `enter` shell all run as the project's user.
- issue    ==> true right now and wants an action: docker down, a TOML that doesn't parse, a container
               with no config left, ... Detected, never stored: fix it and it is gone.
- error    ==> something that failed once: a log. A tab in the TUI; a CLI run prints its own at the end.
- TUI      ==> no argument. One tab per kind of thing, refreshed every 2s. i/k up/down, j/l or 1-9 for
               tabs, g/G top/bottom, q quits; the bottom line has the keys of the tab you are on.
──────────────────────────────────────────────
- containerctl                          ==> launch TUI
- containerctl help                     ==> show this help
- containerctl help full                ==> the why and the concepts, then this help (= the README)
- containerctl help config              ==> example project, service, image and secrets TOML
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
- containerctl exec <id> <cmd...>       ==> execute command in container (exits with its code)
- containerctl shell <id>               ==> open shell in container
- containerctl shell <id> -s /bin/bash  ==> open bash in container
──────────────────────────────────────────────
- containerctl enter <project>          ==> enter project (spawn or attach)
- containerctl enter <proj> user=root   ==> enter project as another user
- containerctl enter <shortcut>         ==> enter project, landing in a [shortcuts] dir
- containerctl spawn <project>          ==> force-create project container
- containerctl destroy <project>        ==> stop and remove project container (checks, then asks)
- containerctl destroy <project> -f     ==> same, without asking (--force; DIRTY is still said)
- containerctl projects                 ==> list configured projects
- containerctl projects -w              ==> wide: full shortcut lists, never clip
──────────────────────────────────────────────
- containerctl check <project>          ==> is the workspace safe to destroy? (exit 1 when DIRTY)
- containerctl check --all              ==> check every configured project
- containerctl check-local              ==> run the same check from inside a container
- containerctl issues                   ==> what wants an action: docker down, broken config, shortcuts, ...
──────────────────────────────────────────────
- containerctl service up <name>               ==> create/start a service
- containerctl service down <name>             ==> stop and remove a service
- containerctl service recreate <name>         ==> recreate a service
- containerctl service scripts <name>          ==> list pre-registered scripts
- containerctl service script <name> <script>  ==> run a pre-registered script
- containerctl services                        ==> list configured services
- containerctl services -w                     ==> wide: never clip columns
──────────────────────────────────────────────
- containerctl image ls                        ==> list recipes: current build, versions (nags past 3)
- containerctl image build <name>              ==> build <repo>:cctl-local-NNNNNN, re-point the recipe tag
- containerctl image check [name]              ==> built image vs recipe, local Dockerfile vs upstream
──────────────────────────────────────────────
Project configs: ~/.config/containerctl/projects/<name>.toml
Service configs: ~/.config/containerctl/services/<name>.toml
Image recipes:   ~/.config/containerctl/images/<name>.toml + <name>.Dockerfile [+ <name>/ build context]
Secrets:         ~/.config/containerctl/secrets.toml (chmod 600; a project's [secrets] picks keys from it)
```
