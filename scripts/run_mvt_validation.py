"""One-off driver for the Phase 2 mvt validation backtest -- loads the
already-cached R1000+SPY 5y price panel (no new network cost) and runs
mvt.compute.run_validation() against the full universe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zenith.cas import store_cas
import pandas as pd
import io

from zenith.mom.mvt import compute as mc

d = store_cas.cache_get("prices_5y", max_age_hours=999999)
px = {}
for t, j in d.items():
    df = pd.read_json(io.StringIO(j), orient="split")
    df.index = pd.to_datetime(df.index)
    px[t] = df

print(f"[run_mvt_validation] loaded {len(px)} tickers from cache")
mc.run_validation(px)
print("[run_mvt_validation] done")
