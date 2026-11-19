# Reaching HPC3 through the office Studio

For the MacBook off campus without the UCI VPN, when Tailscale still reaches the office Mac Studio
(`studio-work`), which is on campus and has its own HPC3 key.

| Script | What it does |
|---|---|
| `h3 "<command>"` | Runs one command on the HPC3 login node (submit, monitor, light inspection; `srun` for anything heavier) |
| `h3pull <HPC3 path> <local dir> [rsync options]` | Small tables and reports: HPC3 transfer node to Studio scratch to MacBook |
| `h3push <HPC3 dir> <local path>...` | Code and configs: MacBook to Studio scratch to HPC3 transfer node |
| `h3wait <seconds> <job id>...` | Waits until none of the jobs is queued or running |

The relay folder on the Studio is `~/LocalDocuments/compute/glymph_relay/` (local disk, not Dropbox, not iCloud).
Large data never goes this way: it stays on the host that computes it.
