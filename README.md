# makefun-free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Rizal9/makefun-free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Rizal9/makefun-free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API over ten simulated free-tier providers. Each request is answered by the best free model that meets its requirements, ranked by the brief's quality table; the load is spread over the providers under their per-trial budgets, a refused provider is failed over to the next candidate, and the server exits 0 when the env says the work is done. No model calls: a deterministic, scripted router — every choice comes from the brief, the requests and the providers' answers, so a live trial and its replay leave the same state export byte for byte.

Practice trials (kit `grade-local --driver process`): 99 on trial 0, 99 on trial 1, 99 on trial 2 — quality 99, served 100, wire 100, failover 100, catalog 100 on each.

## Run it

```sh
MAKEFUN_PROMPT_FILE=<brief> MAKEFUN_ENV_URL=<env> node main.ts
```

`node main.ts --list` prints the catalog the server registers, and exits 0.

---

<sub>Entered on [make.fun](https://make.fun/?ref=abmjq4f4&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
