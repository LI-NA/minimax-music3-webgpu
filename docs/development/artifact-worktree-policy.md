# Artifact Worktree Policy

`artifacts/` in the main checkout is canonical. Linked worktrees must use a Windows directory junction to it and must never copy the artifact tree.

Run converter download and build commands only from the main checkout, one at a time.

Before replacing an existing release, preserve the complete release directory and its matching receipt under `artifacts/archive/`. Verify hardlinks or copied files before promotion. Archived generations are immutable and must never be removed automatically. Successful `music-variable` replacements store the prior release and receipt as one recovery unit under a unique `artifacts/archive/music-variable/<generation>/` directory. Successful legacy `music-5s` replacements preserve the prior release, without a receipt, under `artifacts/archive/music-5s/<generation>/release`.

In a linked worktree, set every applicable `MINIMAX_*_CHROME_PROFILE` to `artifacts/worktree-profiles/<worktree-name>/<suite>`. Do not use a default or shared profile.

Before removing a worktree, verify and delete only its junction without recursion. Never recursively delete through the junction. Remove its namespaced profile separately after Chrome exits.
