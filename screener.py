"""S&P500 스크리너 v3

"테슬라 가격 패턴" 종목 발굴: 장기 우상향 골격 위에서 급락-회복 사이클을
반복하는 종목(사이클형)이 급락 국면에 들어왔을 때 포착한다.

조건 1 — 시장의 지속적 관심: 2년 평균 거래대금 상위 10%
조건 2 — 사이클 이력: -25% 초과 낙폭 후 전고점 완전 회복 완주 2회 이상,
         최근 5년 총수익률 > 0 (나락형 제외)
조건 3 — 현재 국면: 랠리(고점이 그 1년 전 대비 +30%)로 만든 최근 2년 내
         고점에서 -25% 이상 급락 중

출력: data.json (대시보드가 fetch)
"""

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

CONFIG = {
    # ── 조건 1: 시장의 지속적 관심 ──────────────────────────────
    "LIQUIDITY_YEARS": 2,          # 거래대금 검증 기간(년)
    "MIN_DOLLAR_VOLUME_M": 500,    # 일평균 거래대금 임계값(백만 달러)
    "SUSTAIN_WINDOW_DAYS": 63,     # 분기(63거래일) 이동평균이 기간 내내 임계값 이상이어야 함
    "INTEREST_RECENT_DAYS": 21,    # 관심도 추세: 최근 1개월(거래일) 평균 거래대금
    # ── 조건 2: 사이클 이력 ────────────────────────────────────
    "CYCLE_LOOKBACK_YEARS": 10,    # 사이클 카운트 대상 기간(년)
    "CYCLE_DRAWDOWN_PCT": 25,      # 사이클로 인정하는 최소 낙폭(%)
    "MIN_COMPLETED_CYCLES": 2,     # 완주(전고점 회복) 최소 횟수
    "LONG_TERM_YEARS": 5,          # 장기 우상향 확인 기간(년)
    "MIN_HISTORY_YEARS": 5,        # 최소 상장(데이터) 기간 — 미달 시 제외
    # ── 조건 3: 상승 후 급락 국면 ──────────────────────────────
    "PEAK_WINDOW_YEARS": 2,        # 기준 고점 탐색 기간(년)
    "PEAK_RUNUP_PCT": 30,          # 고점이 그 시점 1년 전 대비 최소 상승률(%)
    "CURRENT_DROP_PCT": 25,        # 고점 대비 현재 최소 하락률(%)
}

TRADING_DAYS_PER_YEAR = 252
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500_members() -> pd.DataFrame:
    """위키백과에서 현재 S&P500 구성 종목(티커/이름/섹터)을 가져온다."""
    resp = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (sp500-screener)"}, timeout=30)
    resp.raise_for_status()
    table = pd.read_html(StringIO(resp.text))[0]
    members = pd.DataFrame({
        "ticker": table["Symbol"].str.replace(".", "-", regex=False),  # BRK.B → BRK-B (yfinance 표기)
        "name": table["Security"],
        "sector": table["GICS Sector"],
    })
    return members.drop_duplicates(subset="ticker").reset_index(drop=True)


def download_history(tickers: list[str], years: int) -> pd.DataFrame:
    """전 종목 일봉(수정종가·거래량)을 일괄 다운로드. 반환: MultiIndex 컬럼 (ticker, field)."""
    df = yf.download(
        tickers,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    return df


def count_completed_cycles(close: pd.Series, min_drawdown_pct: float) -> int:
    """낙폭 곡선(직전 최고가 대비)에서 '-min_drawdown% 초과 하락 → 낙폭 0 복귀' 완주 횟수.

    복귀 전에 낙폭이 여러 번 임계값을 넘나들어도 신고가 복귀 시점에 1회로 센다.
    진행 중(미회복) 낙폭은 세지 않는다.
    """
    drawdown = close / close.cummax() - 1.0
    threshold = -min_drawdown_pct / 100.0
    completed = 0
    below = False
    for dd in drawdown.to_numpy():
        if dd <= threshold:
            below = True
        elif dd >= -1e-12 and below:  # 전고점 완전 회복
            completed += 1
            below = False
    return completed


def analyze_ticker(close: pd.Series, cfg: dict) -> dict | None:
    """조건 2·3에 필요한 지표 계산. 데이터 기간 미달이면 None."""
    close = close.dropna()
    if close.empty:
        return None
    today = close.index[-1]
    history_years = (today - close.index[0]).days / 365.25
    if history_years < cfg["MIN_HISTORY_YEARS"]:
        return None

    # ── 조건 2: 사이클 이력 ──
    cycles = count_completed_cycles(close, cfg["CYCLE_DRAWDOWN_PCT"])
    five_years_ago = today - timedelta(days=round(cfg["LONG_TERM_YEARS"] * 365.25))
    base = close.loc[:five_years_ago]
    long_term_return = float(close.iloc[-1] / base.iloc[-1] - 1.0) * 100 if not base.empty else None
    cond2 = cycles >= cfg["MIN_COMPLETED_CYCLES"] and long_term_return is not None and long_term_return > 0

    # ── 조건 3: 상승 후 급락 ──
    window = close.loc[today - timedelta(days=round(cfg["PEAK_WINDOW_YEARS"] * 365.25)):]
    peak_price = window.max()
    peak_date = window.idxmax()
    pre_peak = close.loc[:peak_date - timedelta(days=365)]
    runup = float(peak_price / pre_peak.iloc[-1] - 1.0) * 100 if not pre_peak.empty else None
    current_drop = float(close.iloc[-1] / peak_price - 1.0) * 100
    cond3 = (
        runup is not None
        and runup >= cfg["PEAK_RUNUP_PCT"]
        and current_drop <= -cfg["CURRENT_DROP_PCT"]
    )

    return {
        "data_start": str(close.index[0].date()),
        "cycles": cycles,
        "long_term_return": round(long_term_return, 1) if long_term_return is not None else None,
        "cond2": bool(cond2),
        "price": round(float(close.iloc[-1]), 2),
        "peak_price": round(float(peak_price), 2),
        "peak_date": str(peak_date.date()),
        "runup": round(runup, 1) if runup is not None else None,
        "current_drop": round(current_drop, 1),
        "cond3": bool(cond3),
    }


def run() -> dict:
    cfg = CONFIG
    members = fetch_sp500_members()
    print(f"S&P500 구성 종목: {len(members)}개")

    data = download_history(members["ticker"].tolist(), cfg["CYCLE_LOOKBACK_YEARS"])
    print("다운로드 완료")

    # ── 조건 1: 분기 이동평균 거래대금이 최근 2년 내내 임계값 이상 ──
    liquidity_days = cfg["LIQUIDITY_YEARS"] * TRADING_DAYS_PER_YEAR
    threshold = cfg["MIN_DOLLAR_VOLUME_M"] * 1e6
    rows = []
    for t in members["ticker"]:
        if t not in data.columns.get_level_values(0):
            continue
        px = data[t]
        dollar_vol = (px["Close"] * px["Volume"]).dropna().tail(liquidity_days)
        if len(dollar_vol) < liquidity_days * 0.8:  # 2년치가 안 되면 "내내 유지" 검증 불가
            continue
        rolling = dollar_vol.rolling(cfg["SUSTAIN_WINDOW_DAYS"]).mean().dropna()
        roll_min = float(rolling.min())
        if roll_min < threshold:
            continue
        avg_2y = float(dollar_vol.mean())
        avg_recent = float(dollar_vol.tail(cfg["INTEREST_RECENT_DAYS"]).mean())
        rows.append({
            "ticker": t,
            "avg_dollar_vol": avg_2y,
            "roll_min": roll_min,
            "interest_ratio": avg_recent / avg_2y,
        })
    top_liquid = pd.DataFrame(rows).sort_values("avg_dollar_vol", ascending=False).reset_index(drop=True)
    print(f"조건 1 통과(분기평균 거래대금 {cfg['LIQUIDITY_YEARS']}년 내내 ${cfg['MIN_DOLLAR_VOLUME_M']}M 이상): {len(top_liquid)}개")

    # ── 조건 2·3 — 조건 1 통과 전 종목의 지표를 계산해 전부 싣는다 (단계별 확인용) ──
    metric_keys = (
        "data_start", "cycles", "long_term_return", "price",
        "peak_price", "peak_date", "runup", "current_drop", "cond2", "cond3",
    )
    meta = members.set_index("ticker")
    stocks, candidates = [], []
    for _, liq_row in top_liquid.iterrows():
        t = liq_row["ticker"]
        try:
            metrics = analyze_ticker(data[t]["Close"], cfg)
        except Exception as e:
            print(f"  [건너뜀] {t}: {e}", file=sys.stderr)
            continue
        entry = {
            "ticker": t,
            "name": meta.loc[t, "name"],
            "sector": meta.loc[t, "sector"],
            "avg_dollar_vol_b": round(liq_row["avg_dollar_vol"] / 1e9, 2),  # 십억 달러
            "roll_min_b": round(liq_row["roll_min"] / 1e9, 2),
            "interest_ratio": round(liq_row["interest_ratio"], 2),
            # 데이터 5년 미만이면 지표 없이 조건 2 탈락 처리
            **(dict.fromkeys(metric_keys) if metrics is None else {k: metrics[k] for k in metric_keys}),
        }
        entry["cond2"] = bool(entry["cond2"])
        entry["cond3"] = bool(entry["cond3"])
        stocks.append(entry)
        if entry["cond2"] and entry["cond3"]:
            candidates.append(entry)

    stocks.sort(key=lambda x: -x["avg_dollar_vol_b"])
    candidates.sort(key=lambda x: x["current_drop"])
    n_pool = sum(1 for s in stocks if s["cond2"])
    print(f"조건 2 통과(관심 풀): {n_pool}개 / 조건 3 통과(오늘의 후보): {len(candidates)}개")

    now_utc = datetime.now(timezone.utc)
    return {
        "updated_utc": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "updated_kst": (now_utc + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M KST"),
        "config": cfg,
        "universe_count": len(members),
        "stocks": stocks,
        "candidates": [c["ticker"] for c in candidates],
    }


if __name__ == "__main__":
    t0 = time.time()
    result = run()
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"data.json 저장 완료 ({time.time() - t0:.0f}초)")
