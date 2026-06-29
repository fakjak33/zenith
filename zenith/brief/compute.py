"""Weekly Brief compute orchestrator.

    python -m zenith.brief.compute

Pulls each section's free data, assembles a single brief.json under data/brief/,
and prints a short status. Every section degrades gracefully.
"""

from __future__ import annotations

from datetime import date

from . import save, DISCLAIMER
from . import sources as src
from ..cas.sources import prices as cas_prices
from ..cas import store_cas
from ..cas.universe import SECTORS, INDUSTRY_ETFS, REGION_SECTOR_ETFS


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:+.1%}"


def _cas_highlights() -> dict:
    frm = [s for s in store_cas.load("factor_rotation", []) if s.get("family") == "frm_composite"]
    frm.sort(key=lambda s: s["signal"], reverse=True)
    cons = store_cas.load("consensus", [])
    hr = store_cas.load("hitrate", {})
    return {"frm_top": frm[:5], "frm_bottom": frm[-5:][::-1],
            "consensus_top": cons[:5],
            "hitrate": hr.get("by_horizon", {})}


def talking_points(ov: list[dict], sectors: list[dict], rt: dict, gx: list[dict],
                   heat: dict) -> list[str]:
    o = {r["ticker"]: r for r in ov}
    pts: list[str] = []
    spy = o.get("SPY", {})
    if spy:
        dirn = "rose" if (spy.get("w1") or 0) >= 0 else "fell"
        pts.append(f"U.S. stocks {dirn} this week — S&P 500 {_pct(spy.get('w1'))} "
                   f"(1-month {_pct(spy.get('m1'))}); Nasdaq {_pct(o.get('QQQ',{}).get('w1'))}, "
                   f"small caps {_pct(o.get('IWM',{}).get('w1'))}.")
    if sectors:
        lead, lag = sectors[0], sectors[-1]
        pts.append(f"Sectors: {lead['label']} led ({_pct(lead['w1'])}), "
                   f"{lag['label']} lagged ({_pct(lag['w1'])}).")
    gld, uso, uup = o.get("GLD", {}), o.get("USO", {}), o.get("UUP", {})
    pts.append(f"Commodities/FX: gold {_pct(gld.get('w1'))}, oil {_pct(uso.get('w1'))}, "
               f"the dollar {_pct(uup.get('w1'))}.")
    curve = rt.get("curve", {})
    ff = rt.get("fed_funds", {})
    if curve.get("DGS10"):
        pts.append(f"Rates: 10y Treasury at {curve['DGS10']['value']}%, "
                   f"2s10s {curve.get('T10Y2Y',{}).get('value','?')}. "
                   f"Fed funds {ff.get('target_lower','?')}–{ff.get('target_upper','?')}%; "
                   f"the 1y market implies ~{ff.get('implied_12m_change_bps','?')} bps over 12m.")
    vix = o.get("^VIX", {})
    if vix:
        mood = "calm" if (vix.get("last") or 99) < 16 else ("elevated" if vix["last"] > 24 else "moderate")
        pts.append(f"Volatility: VIX at {vix.get('last')} ({mood}), 1-week {_pct(vix.get('w1'))}.")
    br = (heat or {}).get("breadth", {})
    if br.get("pct_above_50dma") is not None:
        pts.append(f"Breadth: {br['pct_above_50dma']:.0%} of the S&P 500 is above its 50-day "
                   f"average ({br.get('pct_above_200dma',0):.0%} above the 200-day).")
    if gx:
        s = next((g for g in gx if g["ticker"] == "SPY"), gx[0])
        pts.append(f"Options: SPY dealer gamma is {s['regime']} — "
                   f"{'dealers cushion moves' if s['net_gex_bn'] >= 0 else 'moves can accelerate'}.")
    return pts


def run() -> dict:
    core = list(src.OVERVIEW) + list(SECTORS) + list(INDUSTRY_ETFS) + list(REGION_SECTOR_ETFS)
    px, pst = cas_prices.get_history(core, period="1y")

    ov = src.market_overview(px)
    sectors = src.sector_perf(px)
    industries = src.industry_perf(px)
    heat = src.stock_heatmap()
    rt = src.rates()
    movers = {r["ticker"] for r in (heat.get("best_1w", []) + heat.get("worst_1w", []))}
    news = src.market_moving_news(movers)
    earn = src.earnings()
    gx = src.gex()
    cas_h = _cas_highlights()
    tp = talking_points(ov, sectors, rt, gx, heat)

    brief = {
        "as_of": date.today().isoformat(),
        "disclaimer": DISCLAIMER,
        "talking_points": tp,
        "market_overview": ov,
        "sectors": sectors,
        "industries": industries,
        "stock_heatmap": heat,
        "rates": rt,
        "earnings": earn,
        "news": news,
        "cas": cas_h,
        "gex": gx,
    }
    save("brief", brief)
    print(f"[brief] {brief['as_of']}: overview={len(ov)} sectors={len(sectors)} "
          f"industries={len(industries)} movers={len(movers)} earnings={len(earn)} "
          f"news={len(news)} gex={len(gx)} talking_points={len(tp)} "
          f"(prices ok={pst.get('ok')})")
    return brief


def main() -> None:
    run()


if __name__ == "__main__":
    main()
