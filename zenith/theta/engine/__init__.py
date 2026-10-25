"""THETA RUN engine: pure game logic (no Streamlit), fully unit-tested.

pricing     Black-Scholes prices and greeks (numpy, no scipy)
strategies  tastytrade-style templates -> legs at the sim date
margin      approximate Reg-T buying-power requirement per structure
book        legs, positions, the account: mark, open, close, roll, settle expiry
clock       advance a day / a week / to the next event, with event detection
bot         the mechanical tastytrade benchmark (45 DTE / 16 delta / 50% / 21 DTE)
scoring     run statistics, XP and achievements
scenario    seeded runs, hidden tickers, scripted challenges
"""
