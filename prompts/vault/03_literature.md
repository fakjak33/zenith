# Phase 03 — Literature review (independent of 01–02)
Read 00_master_brief.md and HANDOFF.md first. Use the anthropic-skills:deep-research skill for
the web research. Verify every citation yourself: authors, year, venue and a working
DOI/SSRN/NBER/journal URL.

Goal: for each topic, establish:
- what the academic and serious practitioner literature says;
- how strong the consensus is;
- where it agrees or disagrees with tastylive's teaching.

Topics (add any that the Phase 02 taxonomy surfaces):
1. Variance/volatility risk premium: existence, size, time variation, crash compensation;
   index vs single stock.
2. Option selling as a portfolio: put-writing, buy-writes/covered calls, strangles, iron
   condors; CBOE benchmark-index studies; drawdowns and critiques.
3. Protective puts and tail hedging: cost vs benefit.
4. Implied vs realized volatility and IV level as a predictor of option returns (the logic
   behind IV-rank conditioning); IV term structure; VIX futures basis/roll.
5. Earnings:
   - implied vs realized earnings moves and IV crush;
   - retail option buying before announcements;
   - straddle returns around earnings;
   - post-earnings drift and continuation of earnings-announcement returns. Tie this to the
     user's "outlier moves keep drifting" example and to the citations already in the
     zenith/pead/__init__.py docstring.
6. 0DTE: pricing, who trades them, retail outcomes, gamma and market impact.
7. Skew/smirk and put-call IV spreads as predictors (zenith EDGE already runs an IV-spread screen).
8. Option momentum and other cross-sectional option-return anomalies.
9. Options trading costs and execution: how costs change backtest conclusions.
10. Sizing and risk: Kelly and fractional Kelly; risk of ruin for short premium; leverage and
    margin calls. Case studies: Feb 2018 short-vol ETP collapse, the 2018 OptionSellers.com
    natural-gas loss, Mar 2020.
11. Management rules (early profit-taking, time exits, rolling, stop-losses on short options):
    do they change expected return, or only reshape the distribution?
12. Mechanics worth knowing: Reg-T vs portfolio margin, SPAN, Section 1256. Source these from
    regulators/brokers; educational, not tax advice.

Starting points believed to exist (verify each; fix or drop any that don't):
- Coval & Shumway (2001); Bakshi & Kapadia (2003); Carr & Wu (2009);
  Bollerslev, Tauchen & Zhou (2009); Goyal & Saretto (2009).
- Ilmanen (2012), "Do financial markets reward buying or selling insurance and lottery tickets?".
- Israelov & Nielsen (2015), "Covered Calls Uncovered"; Israelov (2019), "Pathetic Protection".
- Whaley (2002) on BXM; Ungar & Moran (2009) and Bondarenko on PUT.
- Gao, Xing & Zhang (2018), straddles around earnings;
  de Silva, Smith & So, "Losing is Optional" (retail options and announcement volatility).
- Bernard & Thomas (1989); Brandt, Kishore, Santa-Clara & Venkatachalam (2008),
  earnings-announcement returns.
- Xing, Zhang & Zhao (2010); Cremers & Weinbaum (2010).
- Heston, Jones, Khorram, Li & Mo, "Option Momentum".
- Muravyev & Pearson (2020), options trading costs.
- Bryzgalova, Pavlova & Sikorskaya (2023), retail options trading.
- 0DTE: Beckmeyer, Branger & Gayda; Bandi, Fusari & Renò; Dim, Eraker & Vilkov.
- Practitioner: Julia Spina, The Unlucky Investor's Guide to Options Trading (tastylive research).

Output data/vault/literature.json → {"topics": [...]}. Per topic:
- consensus (2–4 sentences, paraphrased);
- tier (A/B/C);
- papers: [{id, authors, year, title, venue, url, finding (one line), sample/period}].
  Paper ids are what other phases cite as lit:<id>;
- agreement_with_tasty (agrees|mixed|contradicts);
- open_debates;
- implications for the cheat cards.

GATE — show:
- a one-page table of topic × consensus × tier × agreement;
- the citations you could not verify.

Then update HANDOFF.md.
