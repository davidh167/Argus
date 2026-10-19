# Research

Evaluations behind Argus's ranking and clustering choices. Each folder is self-contained: scripts import
from `app/` read-only, write derived data to a gitignored `data/` folder (it comes from personal
postings and is regenerable), and commit only aggregate `reports/`. Extra dependencies are in the
`research` Poetry group (`poetry install --with research`).

| Folder | Question |
|---|---|
| [ranking-eval](ranking-eval/README.md) | Which scorer ranks the roles a user deliberately picked above the rest of the corpus? |
| [cluster-quality](cluster-quality/README.md) | How should the skill phrases extracted from postings be clustered? |
| [stage2-prompt](stage2-prompt/README.md) | Can the triage prompt emit cleaner, more consistent skill names without changing verdicts? |
| [skill-judgment](skill-judgment/README.md) | Can retrieval plus an LLM judge decide whether a profile demonstrates a skill? |

The harnesses read one user's postings and choices, so sample sizes are small (tens of queries) and
every report states its confidence intervals and caveats. The C++/Rust retrieval work in
[`native/`](../native/README.md) is evaluated with the ranking harness.
