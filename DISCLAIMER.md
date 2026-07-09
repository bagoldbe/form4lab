# Disclaimer

**Nothing in form4lab is investment advice.** This is educational research software for studying SEC Form 4 insider-trading data and backtesting mechanics. It is not a recommendation to buy, sell, or hold any security, and using it creates no fiduciary relationship between you and its authors or contributors. If you need investment, legal, or tax advice, get it from a licensed professional — not from a GitHub repository.

## Example strategies

The strategies shipped in `form4lab/strategies/` (`cluster_buy`, `big_exec_buy`, `opportunistic_first_buy`) are pedagogical illustrations of heuristics described in published academic literature (see each module's docstring for its citation). They are:

- **Unvalidated.** Their thresholds are labelled illustrative defaults in the code itself, not the product of any tuning or out-of-sample study in this repo.
- **Expected to underperform after costs.** Nothing in the shipped defaults accounts for commissions, borrow costs, market impact, or taxes; a naive textbook heuristic that looks fine gross of costs routinely doesn't survive contact with them.
- **Not a starting point for your capital.** They exist to demonstrate the `Strategy` interface (see `docs/strategy-authoring.md`), not to be deployed as-is.

## No warranty

form4lab is licensed under the Apache License, Version 2.0. Per §7 (Disclaimer of Warranty) and §8 (Limitation of Liability) of that license, it is provided "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, and its authors and contributors are not liable for any damages arising from its use. See [LICENSE](LICENSE) for the full text.

Backtests in particular embed simplifying assumptions: the simulator fills at end-of-day prices, models slippage/market impact only if you explicitly configure it (off by default), and inherits whatever survivorship bias exists in the ticker universe you point it at. Nothing about a backtested or simulated result is a promise about the future. **Past performance — and simulated or backtested performance even more so — is not indicative of future results.**

## Trading risk

Paper trading (`ALPACA_PAPER=true`) is the default whenever Alpaca execution is enabled, and Alpaca execution itself is off by default (`ALPACA_ENABLED=false`) — form4lab never places an order unless you explicitly opt in. If you turn on live execution (`ALPACA_ENABLED=true`, `ALPACA_PAPER=false`), you are risking real money on a strategy that this repository does not warrant to be profitable, correct, or safe, entirely at your own risk. form4lab is an independent, unaffiliated project — it is not affiliated with, endorsed by, or sponsored by Alpaca Securities LLC or Alpaca Markets; "Alpaca" here refers only to the brokerage API this software can optionally integrate with.

## SEC EDGAR access

form4lab fetches Form 4 filings directly from SEC EDGAR. EDGAR's fair-access policy requires every automated requester to identify itself with a descriptive `User-Agent`; form4lab sends whatever you set `SEC_IDENTITY` to, verbatim, on every request (see `form4lab/data/sec_fetcher.py`) — set it to your own name/org and a real contact email, not a placeholder and not someone else's identity. Filed data is reproduced as-filed and may contain filer errors or omissions; form4lab does not audit, verify, or correct it. form4lab is an independent project and is not affiliated with, endorsed by, or sponsored by the U.S. Securities and Exchange Commission.
