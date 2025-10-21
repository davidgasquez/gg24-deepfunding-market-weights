# GG24 Deep Funding Market Weights 🏗️

This repository contains the code and artifacts used on the initial weights selection for the [Developer Tooling & Infrastructure Deep Funding Round](https://gitcoin.notion.site/GG24-Developer-Tooling-and-Infrastructure-Deep-Funding-Round-286f3309710d806bb97dfe25778f2afe) in GG24.

This method uses [Arbitron](https://github.com/davidgasquez/arbitron) to [rank and score](https://davidgasquez.com/ranking-with-agents/) each repository was selected into the round. It works like this:

1. The [candidate repositories](data/candidate_repositories.csv) are slighly expanded with [some metadata](generate_repository_context.py)
2. A few artificial jurors are created. All with the task of choosing which repository had a larger impact on Ethereum.
3. A competition is run where jurors choose the winner on more than 50000 random pairwise comparisons.
4. A rank and relative weight is derived from the comparisons.
5. Done!

## 🚀 Quickstart

Although the approach is not fully reproducible (most LLM providers don't have a way to set seeds), you can replicate this setup and the weights should converge to similar values. You'll need a couple of things:

- A working Python setup. The easies way to do that is [using `uv`](https://docs.astral.sh/uv/).
- Proper environment secrets. Set `OPENAI_API_KEY` and other keys (e.g: `GEMINI_API_KEY` or `ANTHROPIC_API_KEY`).

If you have that, you should be able to do something like this:

```
uv run --env-file .env duel.py
```

It'll generate a run file inside `data/runs`. You can then do `make weights` to apply multiple methods to derive weights from those comparisons.

The previous commands will use the [snapshoted repository context](data/repository_context.json). If you want to refresh that, run `uv run generate_repository_context.py data/candidate_repositories.csv --out data/repository_context.json`.

## 📊 Data

Some quick stats on the competition I've run.

- Made **53716 comparisons** across **201 unique repositories** using **6 jurors**
- Spent **~100 USD in API credits**
- Comparisons by juror:
  - `dev` made 25024 comparisons
  - `meta` made 6600 comparisons
  - `builder` made 6100 comparisons
  - `senior-dev` made 5695 comparisons
  - `founder` made 5600 comparisons
  - `ethereum-dev` made 4697 comparisons
- The **most compared** repository was compared **655 times**, while the **least compared** one **431 times**.
- The repository with the **largest winrate** was `flashbots/mev-boost` with 512 wins out of 518 comparisons! A 98.8% win rate.
