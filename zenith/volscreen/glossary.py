"""One source for every metric's label, hover help, formula and citation.

Feeds the dataframe column help, the `?` badges and the LEARN section, so a
definition is written once and reads the same everywhere.
"""

from __future__ import annotations

# key: (column label, one-line help, LaTeX formula or "", citation)
METRICS: dict[str, tuple[str, str, str, str]] = {
    "stars": ("LIQ ★", "Liquidity rating 1-4 from option volume, open interest, ATM / 25Δ spread and strike "
              "density (tastytrade-style, our thresholds).", "", "Zenith thresholds (config.VOLSCREEN_LIQ)"),
    "class": ("CLASS", "CORE = consistently liquid; SURGE = volume far above its own norm (often an event); "
              "FADING = volume well below norm; THIN = avoid.", "", ""),
    "opt_vol": ("OPT VOL", "Contracts traded today across every listed expiry.", "", "Cboe delayed quotes"),
    "opt_vol20": ("VOL 20D", "Average daily option volume over the last 20 observed sessions.", "", ""),
    "oi": ("OPEN INT", "Total open interest across the chain.", "", ""),
    "atm_spread": ("ATM SPRD", "(ask - bid) / mid of the at-the-money options in the ~30 DTE expiry: "
                   "what a round trip costs.", r"\text{spread} = \frac{a - b}{(a + b)/2}", ""),
    "d25_spread": ("25Δ SPRD", "Relative spread at the 25-delta wings, where short strangles live.", "", ""),
    "iv30": ("IV30", "30-day constant-maturity ATM implied vol, interpolated in total variance.",
             r"\sigma_{30}^2 \tfrac{30}{365} = w(T_1) + \tfrac{T - T_1}{T_2 - T_1}\,(w(T_2) - w(T_1))",
             "Cboe VIX methodology (variance interpolation)"),
    "ivx30": ("IVx30", "30-day IV with the earnings event variance removed (ex-earnings).",
              r"\sigma_{ex}^2 = \frac{w_3 - w_2}{T_3 - T_2}, \quad \varepsilon^2 = w_2 - \sigma_{ex}^2 T_2", ""),
    "front": ("FRONT", "ATM IV of the first listed expiry at least 7 days out.", "", ""),
    "back": ("BACK", "ATM IV of the listed expiry nearest 90 days.", "", ""),
    "ratio": ("30/90", "IV30 / IV90. Below 1 = contango (normal, calm); above 1 = backwardation "
              "(near-term stress or an event).", r"\text{ratio} = \sigma_{30} / \sigma_{90}",
              "Johnson (2017); Simon & Campasano (2014)"),
    "ivr": ("IVR", "IV rank: where IV30 sits in its trailing-year min-max range (0-100).",
            r"\text{IVR} = \frac{\sigma - \min_{252}}{\max_{252} - \min_{252}}", "tastytrade convention"),
    "ivp": ("IVP", "IV percentile: share of the trailing year's days with IV30 below today's.", "", ""),
    "iv_z": ("σ-Z", "Today's IV in standard deviations of its own trailing-year distribution, per tenor.",
             r"z_T = \frac{\sigma_T - \bar\sigma_T}{s_T}", ""),
    "yz20": ("RV YZ20", "Yang-Zhang 20-day realized vol: overnight + open-to-close + Rogers-Satchell, "
             "minimum-variance and gap-aware.",
             r"\sigma^2_{YZ} = \sigma^2_o + k\sigma^2_c + (1-k)\sigma^2_{RS},\; k = \frac{0.34}{1.34 + \frac{n+1}{n-1}}",
             "Yang & Zhang (2000)"),
    "pk20": ("RV PK20", "Parkinson high-low range estimator (ignores gaps).",
             r"\sigma^2_P = \frac{1}{4\ln 2}\,\overline{\ln^2(H/L)}", "Parkinson (1980)"),
    "gk20": ("RV GK20", "Garman-Klass: range plus open/close.",
             r"\sigma^2_{GK} = \overline{\tfrac12 \ln^2(H/L) - (2\ln2 - 1)\ln^2(C/O)}", "Garman & Klass (1980)"),
    "rs20": ("RV RS20", "Rogers-Satchell: unbiased under drift.",
             r"\sigma^2_{RS} = \overline{\ln\tfrac{H}{C}\ln\tfrac{H}{O} + \ln\tfrac{L}{C}\ln\tfrac{L}{O}}",
             "Rogers & Satchell (1991)"),
    "cc20": ("RV CC20", "Close-to-close realized vol (zero-mean).", r"\sigma^2 = 252\,\overline{r_t^2}", ""),
    "ewma": ("EWMA", "RiskMetrics exponentially weighted vol, λ = 0.94.",
             r"\sigma^2_t = \lambda \sigma^2_{t-1} + (1-\lambda) r^2_{t-1}", "J.P. Morgan RiskMetrics (1996)"),
    "har30": ("HAR30", "HAR-RV forecast of the next 21 trading days from daily / weekly / monthly "
              "realized variance.", r"RV_{t+h} = \beta_0 + \beta_d RV_t + \beta_w RV^{(5)}_t + \beta_m RV^{(22)}_t",
              "Corsi (2009)"),
    "garch30": ("GARCH30", "GARCH(1,1) forecast of average vol over the next 30 days.",
                r"\sigma^2_t = \omega + \alpha r^2_{t-1} + \gamma r^2_{t-1}\mathbb 1_{r<0} + \beta\sigma^2_{t-1}",
                "Bollerslev (1986); Glosten, Jagannathan & Runkle (1993)"),
    "vrp30": ("VRP30", "Ex-earnings IV30 minus the GARCH forecast of 30-day realized vol, in vol points. "
              "Positive = options priced above expected movement.", r"VRP_T = \sigma^{ex}_T - E_t[RV_{t,t+T}]",
              "Carr & Wu (2009); Bollerslev, Tauchen & Zhou (2009)"),
    "iv_rv": ("IV/RV", "IV30 divided by Yang-Zhang 20-day realized vol.", "", ""),
    "gs": ("HV-IV", "Goyal-Saretto spread ln(HV252) - ln(IV30): very negative = IV far above history.",
           r"\ln \sigma^{HV}_{252} - \ln \sigma^{IV}_{30}", "Goyal & Saretto (2009)"),
    "vrp_z": ("VRP Z", "Today's VRP30 in standard deviations of its own history.", "", ""),
    "slope5": ("SLOPE5", "OLS slope of VRP30 over the last 5 observations, vol points per day: the alarm.",
               r"\hat b = \frac{\sum (x - \bar x)(v - \bar v)}{\sum (x - \bar x)^2}", ""),
    "state": ("STATE", "VRP state machine: ALL-CLEAR / ALARM / CONFIRMED / RECOVERING (practitioner "
              "heuristic: slope = alarm, VRP = confirmation, slope direction = all-clear).", "", ""),
    "skew25": ("SKEW25", "25Δ put IV minus 25Δ call IV at 30 days: how rich the downside is.",
               r"\sigma_{25P} - \sigma_{25C}", ""),
    "em30": ("EM30", "One-standard-deviation 30-day expected move in price.", r"S\,\sigma_{30}\sqrt{30/365}", ""),
    "earn_move": ("EARN MOVE", "Market-implied earnings move from the term-structure kink.",
                  r"E|\Delta| = \varepsilon\sqrt{2/\pi}", ""),
    "score": ("SCORE", "Checklist score 0-100 (pass = full weight, warn = half).", "", ""),
}


def label(key: str) -> str:
    return METRICS[key][0]


def help(key: str) -> str:
    return METRICS[key][1]


def help_map() -> dict[str, str]:
    """column label -> help text, for ui_charts.colcfg."""
    return {v[0]: v[1] for v in METRICS.values()}


REFERENCES = [
    ("Parkinson, M. (1980)", "The extreme value method for estimating the variance of the rate of return. "
     "Journal of Business 53(1)."),
    ("Garman, M. & Klass, M. (1980)", "On the estimation of security price volatilities from historical data. "
     "Journal of Business 53(1)."),
    ("Rogers, L. & Satchell, S. (1991)", "Estimating variance from high, low and closing prices. "
     "Annals of Applied Probability 1(4)."),
    ("Yang, D. & Zhang, Q. (2000)", "Drift-independent volatility estimation based on high, low, open and "
     "close prices. Journal of Business 73(3)."),
    ("Bollerslev, T. (1986)", "Generalized autoregressive conditional heteroskedasticity. "
     "Journal of Econometrics 31(3)."),
    ("Glosten, Jagannathan & Runkle (1993)", "On the relation between the expected value and the volatility "
     "of the nominal excess return on stocks. Journal of Finance 48(5)."),
    ("Corsi, F. (2009)", "A simple approximate long-memory model of realized volatility. "
     "Journal of Financial Econometrics 7(2)."),
    ("Carr, P. & Wu, L. (2009)", "Variance risk premiums. Review of Financial Studies 22(3)."),
    ("Bollerslev, Tauchen & Zhou (2009)", "Expected stock returns and variance risk premia. "
     "Review of Financial Studies 22(11)."),
    ("Goyal, A. & Saretto, A. (2009)", "Cross-section of option returns and volatility. "
     "Journal of Financial Economics 94(2)."),
    ("Johnson, T. (2017)", "Risk premia and the VIX term structure. "
     "Journal of Financial and Quantitative Analysis 52(6)."),
    ("Simon, D. & Campasano, J. (2014)", "The VIX futures basis: evidence and trading strategies. "
     "Journal of Derivatives 21(3)."),
    ("Vasquez, A. (2017)", "Equity volatility term structures and the cross-section of option returns. "
     "Journal of Financial and Quantitative Analysis 52(6)."),
    ("Ledoit, O. & Wolf, M. (2004)", "Honey, I shrunk the sample covariance matrix. "
     "Journal of Portfolio Management 30(4)."),
]
