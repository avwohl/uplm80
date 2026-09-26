# `git stash` / `git stash pop` across worktrees

A `git stash` followed by `git stash pop`, run in a worktree with nothing to
save, pops whatever another worktree of the same repository stashed last.
It is ordinary git behaviour, repeatable every time, and it bit a workflow
agent working on uplm80 0.4.0.

## What happened (2026-09-25)

- **Where:** worktree `~/src/uplm80-abi` (the 0.4.0 calling-convention
  work), workflow `wf_7e0cf5a3-207`, agent `ac2fdfdd29b2d6ad3`, session
  `ce44c1eb-5918-40e2-82fe-76c2055cfb13`, at 2026-09-25T17:19:30Z.
- **Why the agent stashed:** only to read `uplm80/codegen.py` as it was at
  commit `c8c4a91`. The worktree had no changes to set aside.
- **The command**, one Bash call: two pylint runs, whose output went to
  `/tmp/claude-lint-new.txt`, and then this, verbatim apart from the
  scratch path:

  ```sh
  git stash -q 2>/dev/null;
  git -C /Users/wohl/src/uplm80-abi show c8c4a91:uplm80/codegen.py > <scratch>/codegen_base.py;
  git stash pop -q 2>/dev/null;
  cat /tmp/claude-lint-new.txt | head -30
  ```

- **What it did:** the `git stash` saved nothing. The `git stash pop` then
  applied `stash@{0}: WIP on fix/mpm-page-zero-relocation: 361bdf8 codegen:
  an MP/M program cannot reach page zero with a literal`, which another
  worktree had made on 2026-09-23. That entry is a 217-line insertion and
  66-line deletion in `uplm80/codegen.py`, and it conflicted with the
  0.4.0 branch, leaving `UU uplm80/codegen.py`.
- **Why it went unnoticed at first:** `-q` and `2>/dev/null` silenced both
  git's "No local changes to save" and the pop's conflict report. The only
  sign in the output was git's "The stash entry is kept in case you need it
  again."
- **Recovery:** the agent ran `git checkout HEAD -- uplm80/codegen.py`,
  checked the file against its own snapshot, and confirmed the stash entry
  was still there. Because the pop conflicted, git kept the entry, so
  nothing was lost.
- **Earlier use of the same pattern:** at 2026-09-25T16:36:34Z the same
  agent ran `git stash -q; pytest ...; git stash pop -q` in
  `~/src/upeepz80-abi` to run tests with and without its change. That
  worktree had changes, so the stash was its own, and no harm resulted.
  It is the same hazard in waiting.

## Why it happens

1. `git stash` with no local changes prints "No local changes to save",
   creates no entry, and **exits 0**. A script cannot tell from the exit
   status that nothing was saved.
2. **`refs/stash` is shared by every worktree of a repository.** The only
   per-worktree refs are `HEAD`, `refs/bisect/*`, `refs/worktree/*` and
   `refs/rewritten/*`.
3. `git stash pop` takes `stash@{0}`, the newest entry, whichever worktree
   or branch made it.

So stash-then-pop is only safe when the stash is known to have created an
entry and no other worktree has pushed one since. Neither holds in a
repository where several agents work in parallel worktrees.

## Reproduction

Verified 2026-09-26 with git 2.50.1 (Apple Git-155).

```sh
#!/bin/sh
# Two worktrees of one repository; A stashes a change, B (clean) does stash/pop.
set -e
D=${1:-$(mktemp -d)}
mkdir -p "$D" && cd "$D"
git init -q repo && cd repo
git config user.email t@example.com && git config user.name t
echo base > f.txt && git add f.txt && git commit -qm base
git worktree add -q ../B -b other          # worktree B, branch other
echo "A's unfinished work" > f.txt          # worktree A (repo/, branch main)
git stash push -q -m "A's work"
echo "A: $(git stash list)"
cd ../B
echo "B before: status='$(git status --short)'"
git stash -q;     echo "B: git stash (clean tree) -> exit $?"
git stash pop -q; echo "B: git stash pop -> exit $?"
echo "B after: status='$(git status --short)' f.txt='$(cat f.txt)'"
echo "stash list now: '$(git stash list)'"
```

Output:

```
A: stash@{0}: On main: A's work
B before: status=''
B: git stash (clean tree) -> exit 0
B: git stash pop -> exit 0
B after: status=' M f.txt' f.txt='A's unfinished work'
stash list now: ''
```

This is worse than the 2026-09-25 incident. There the pop conflicted, so
git kept the entry. Here it applies cleanly: A's saved work moves into B,
the entry is dropped from the shared list, and both commands report
success. A's owner finds the stash gone, and B's owner has changes they
never made.

## What to do instead

Each use of stash has a direct replacement that never touches the shared
list:

- **Read an old version of a file:** `git show <rev>:<path> > somefile`.
  This is all the 2026-09-25 agent needed.
- **Run tests with and without a change:** make a throwaway worktree from
  a clean HEAD, test there, and remove it. The edited tree is never
  touched, so there is nothing to restore if the run is interrupted:

  ```sh
  git worktree add --detach <scratch>/clean HEAD
  (cd <scratch>/clean && pytest ...)
  git worktree remove <scratch>/clean
  ```

- **Set work aside to switch branches:** with worktrees there is no need
  to switch; open another worktree. If a branch really must change, commit
  the work as `wip` on its own branch and undo it later with
  `git reset --soft HEAD~1`. A commit on a branch cannot be picked up from
  elsewhere.
- **Never discard stderr of a git command that changes files.** That is
  what hid the 2026-09-25 failure.

### A private stash, when one is really needed

`git stash create` makes a stash commit without putting it on the shared
list, and prints nothing when there is nothing to save. Kept under
`refs/worktree/`, it is visible only to the worktree that made it:

```sh
sha=$(git stash create)                       # empty if nothing to save
[ -n "$sha" ] && git update-ref refs/worktree/wip "$sha" && git reset -q --hard
...                                           # work on the clean tree
git stash apply -q --index refs/worktree/wip && git update-ref -d refs/worktree/wip
```

Verified 2026-09-26 in the same two-worktree setup:

```
clean tree: stash create -> ''
set aside: f.txt='base', shared list='stash@{0}: On main: A's work'
A sees B's slot? no
restored: f.txt='B's work', shared list='stash@{0}: On main: A's work'
```

Its limits:

- `git stash create` does not save untracked files.
- There is one slot per worktree unless each use gets its own ref name.
- `git stash drop` refuses a commit ID ("is not a stash reference"), so the
  slot is removed with `git update-ref -d`, as above.

### Why "use stash carefully" is not enough

Labelling the entry (`git stash push -m <tag>`) and popping it by that
label still has a race. Entries are numbered from the top, so if another
worktree pushes between the lookup and the pop, `stash@{n}` names a
different entry. The private slot above has no race.

## State left behind

- The colliding entry was dropped on 2026-09-26. It was `601a9b2`, `WIP on
  fix/mpm-page-zero-relocation: 361bdf8`, made 2026-09-23 06:55 (-0400),
  and it held only unstaged changes to `uplm80/codegen.py` (+217/-66;
  nothing staged, no untracked files). That file was byte for byte the
  `uplm80/codegen.py` of commit `995bbc6` ("codegen: five ways a
  declaration could come out wrong"), made two minutes later and in
  `origin/main` (same blob, `c14c41b`), so nothing in it was lost. Until
  git's garbage collection prunes it, `git stash store 601a9b2` would put
  it back.
- No other repository has a stash entry: `upeepz80`, `um80_and_friends`,
  `mpm2`, `cpmemu`, `romwbw_emu` and `80un` each list none (2026-09-26).

## Follow-ups

- A bug report was drafted on 2026-09-26. It is queued locally, and
  `/feedback` shows and sends it.
- A rule in `~/.claude/CLAUDE.md` (no plain `git stash` or `git stash
  pop`; use the replacements above; never discard a file-changing git
  command's stderr) and git aliases `git wstash` / `git wpop` wrapping the
  private slot were proposed. The owner is reviewing the issue; no rule
  has been added.
