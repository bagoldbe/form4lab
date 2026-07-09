# Contributing

Thanks for considering a contribution. form4lab is a maintained, solo side project (see [SUPPORT.md](SUPPORT.md)) — there's no team and no roadmap, but useful pull requests are welcome.

## Scope

**Welcome:**
- The data pipeline — SEC EDGAR ingestion, Form 4 parsing, rate-limit handling.
- The backtester / simulation engine mechanics.
- Documentation.
- Bug fixes anywhere in the codebase.

**Out of scope:** trading strategies, alpha, or signal tuning. The strategies shipped in `form4lab/strategies/` are pedagogical illustrations of published heuristics, not tuned or validated for live use — see [DISCLAIMER.md](DISCLAIMER.md). form4lab deliberately ships no validated trading edge, and this project isn't the place to build one. If you want your own strategy, write one against the `Strategy` interface (see [docs/strategy-authoring.md](docs/strategy-authoring.md)) and keep it in your own module — you don't need a merged PR to do that.

## Dev setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[trade,dev]"

export SEC_IDENTITY="Your Name you@example.com"   # required — see "SEC EDGAR access" below
form4lab init-db
```

Run the tests:

```bash
SEC_IDENTITY=x .venv/bin/python -m pytest tests/ -q -m "not live"
```

`-m "not live"` skips the handful of tests that hit real external APIs (see `pytest.ini`).

## Before you push

1. Run the tests (above) and make sure they pass.
2. Stage everything, then run the leak audit against your full history:
   ```bash
   git add -A
   python scripts/leak_audit.py --history
   ```
   Stage first — the leak audit only scans git-tracked files, so an untracked file is invisible to it and won't be checked. Both the tests and the audit must pass before you push.

## SEC EDGAR access

SEC EDGAR's fair-access policy requires every automated requester to identify itself with a descriptive `User-Agent`. Set your own `SEC_IDENTITY` (your name/org + a real contact email) when developing or testing against real EDGAR — never a placeholder and never someone else's identity. See [DISCLAIMER.md](DISCLAIMER.md) for the full note.

## License

By contributing, you agree that your contribution is licensed under the Apache License, Version 2.0 — the same license as the rest of the project (see [LICENSE](LICENSE)). Inbound = outbound; no separate contributor agreement.
