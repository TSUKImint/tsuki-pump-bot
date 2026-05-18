# tsuki-pump-bot — v0.2 plan + research notes

Copied from `/home/ubuntu/PUMPFUN_PLAN.md` for in-repo traceability.

> The brutal-honesty section, the 6-filter rationale, and the cross-source
> citations are kept verbatim. If anything in `README.md` conflicts with
> this document, this document wins.

## Brutal headline

* 87% of Solana memecoins lose >90% of peak in 24h.
* 0.3% of wallets have ever realized $10K+ profit.
* The edge of those 0.3% is *structural* (bot speed, insider intent), not analytical.
* Promising "an actual edge over supercomputer bots" with a Ryzen 7 + 32GB
  + integrated GPU would be a lie. **Selection, not speed, is the only
  honest play for this hardware.**

## Why these 6 filters?

We picked filters that satisfy three constraints simultaneously:

1. **Not speed-bound.** Each filter operates on minutes-to-hours data, not
   nanoseconds. A laptop running them well beats a colo box running raw
   sniper logic, *if the filters genuinely select better tokens.*
2. **Under-exploited.** Each is sold separately as a paid SaaS (Bubblemaps,
   Cielo, Birdeye, XHuntr, SolSniffer, …). Nobody combines them with the
   discipline of a paper-first, weight-tuned, kill-switched bot.
3. **Composable.** Hard-rejects (dev blacklist, bundle concentration) gate
   the entry decision; positive signals (KOL touch, convergence, curve
   prediction, CTO) stack into a 0-100 composite. Each filter can be
   disabled individually to ablate its contribution.

## Sources (English + Chinese + Korean + Russian, 2025-2026)

Embedded in code comments and the original `/home/ubuntu/PUMPFUN_PLAN.md`
that was attached to the user-facing plan message. ~25 sources total
including ChainCatcher, MadeOnSol's 491k-trade backtest, Coinlive's "MEME
hype rollback" piece, AllenHark's public scammer DB, Solana Radar Feb 2026,
QuickNode's pump.fun bot guide, Bullrank's bonding-curve write-up, and
CSDN/Weibo posts on "二次发酵" (second-fermentation = CTO).

## What we will NOT build

Hard no on:

* **Wallet-topology obfuscation** (many-to-many transfers to defeat
  Arkham/Bubblemaps clustering). This is the rug-puller toolkit. Even if
  it's the literal answer to "edge", building it would make this bot
  indistinguishable from a scam-tooling vendor.
* **Stealth bundlers that fake Bubblemaps distribution** for the dev's own
  launches.
* **Volume/holder/bump bots** to game pump.fun's trending algorithm.

Anti-snipe via legitimate Jito bundles for the user's own tokens is in
scope (v0.3+) — that's the same primitive used by professional market
makers to protect their seed orders. It does NOT mimic concentration.

## Mode progression

| Stage | Mode | Duration | Decision point |
|-------|------|----------|----------------|
| 1 | `paper-mock` | a few minutes | smoke-test the plumbing |
| 2 | `paper` | 2-4 weeks | does the filter stack actually pick winners? |
| 3 | `devnet` | 1 week | does the executor work end-to-end? |
| 4 | `mainnet` | hard caps | only after stages 2 & 3 produce convincing forward-test data |

Skipping stage 2 is the single biggest mistake we can make. The whole
premise is "selection, not speed" — if selection doesn't show edge in
forward test, no amount of execution polish saves us.
