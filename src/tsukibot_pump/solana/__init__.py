"""Solana primitives: RPC client, wallet model, pump.fun program math, Jito.

Read-only modules (`rpc`, `pump_program`, `bonding_curve`) work with just
httpx — no `solders` / `solana` SDK dependency, so paper modes can run on
any machine. Write-side modules (`wallet`, `jito_bundle`) import the SDK
lazily and are only loaded in `devnet` / `mainnet` modes.
"""
