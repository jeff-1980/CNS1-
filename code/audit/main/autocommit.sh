#!/usr/bin/env bash
# 后台自动提交：每隔 INTERVAL 秒检查审计结果目录，有新文件或改动就 commit + push 到审计分支，
# 不依赖电脑上的会话醒着。只提交 results/main_objects/ 和 results/DEVIATIONS_AUDIT.md，其他路径一概不碰。
# 有其他 git 操作在进行（.git/index.lock 存在）时跳过本轮；push 失败按 2/4/8/16 秒退避重试。
# 启动（WSL）：nohup setsid bash code/audit/main/autocommit.sh > ~/audit_ext/logs/autocommit.log 2>&1 &
# 查看：tail ~/audit_ext/logs/autocommit.log；停止：kill $(cat ~/audit_ext/logs/autocommit.pid)
set -u
REPO=${REPO:-$HOME/audit_cns}; BRANCH=${BRANCH:-claude/project-thread-ogrtsz}; INTERVAL=${INTERVAL:-600}
PATHS=(results/main_objects results/DEVIATIONS_AUDIT.md)
mkdir -p "$HOME/audit_ext/logs"; echo $$ > "$HOME/audit_ext/logs/autocommit.pid"
log() { echo "$(date -u +%FT%TZ) $*"; }
log "start pid $$ repo $REPO branch $BRANCH interval ${INTERVAL}s"
while true; do
  cd "$REPO" || { log "repo missing"; sleep "$INTERVAL"; continue; }
  if [ -e .git/index.lock ]; then log "git busy, skip"; sleep "$INTERVAL"; continue; fi
  if [ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]; then log "not on $BRANCH, skip"; sleep "$INTERVAL"; continue; fi
  P=(); for x in "${PATHS[@]}"; do [ -e "$x" ] && P+=("$x"); done
  if [ ${#P[@]} -gt 0 ] && [ -n "$(git status --porcelain -- "${P[@]}")" ]; then
    git add -- "${P[@]}"
    files=$(git diff --cached --name-only -- "${P[@]}" | wc -l)
    if [ "$files" -gt 0 ]; then
      git commit -q -m "audit main phase: auto-commit of results snapshot ($files files)" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014jfJVK9Di2khTdmMRyXwzf" -- "${P[@]}" && log "committed $files files"
    fi
  fi
  if [ -n "$(git log --oneline "origin/$BRANCH..HEAD" 2>/dev/null)" ]; then
    for w in 0 2 4 8 16; do
      sleep "$w"
      git pull -q --no-rebase origin "$BRANCH" && git push -q -u origin "$BRANCH" && { log "pushed $(git rev-parse --short HEAD)"; break; }
      [ -e .git/MERGE_HEAD ] && { git merge --abort; log "merge conflict, aborted; will retry next round"; break; }
      log "push failed, retry"
    done
  fi
  sleep "$INTERVAL"
done
