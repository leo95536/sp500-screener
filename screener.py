"""S&P500 스크리너

조건 1 — 시장의 지속적 관심:
분기(63거래일) 이동평균 거래대금(종가×거래량)이 최근 2년 내내 임계값 이상.
거래량(돈의 유입)이 없으면 주가가 뜰 일이 없다는 전제의 1단계 필터.

출력: data.json (대시보드가 fetch)
"""

import json
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

CONFIG = {
    "LIQUIDITY_YEARS": 2,          # 거래대금 검증 기간(년)
    "MIN_DOLLAR_VOLUME_M": 500,    # 일평균 거래대금 임계값(백만 달러)
    "SUSTAIN_WINDOW_DAYS": 63,     # 분기(63거래일) 이동평균이 기간 내내 임계값 이상이어야 함
    "INTEREST_RECENT_DAYS": 21,    # 관심도 추세: 최근 1개월(거래일) 평균 거래대금
    "CHART_YEARS": 10,             # 종목별 차트 제공 기간(년)
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


def run() -> dict:
    cfg = CONFIG
    members = fetch_sp500_members()
    print(f"S&P500 구성 종목: {len(members)}개")

    data = yf.download(
        members["ticker"].tolist(),
        period=f"{max(cfg['CHART_YEARS'], cfg['LIQUIDITY_YEARS'])}y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    print("다운로드 완료")

    liquidity_days = cfg["LIQUIDITY_YEARS"] * TRADING_DAYS_PER_YEAR
    threshold = cfg["MIN_DOLLAR_VOLUME_M"] * 1e6
    meta = members.set_index("ticker")
    stocks = []
    for t in members["ticker"]:
        if t not in data.columns.get_level_values(0):
            continue
        px = data[t]
        close = px["Close"].dropna()
        dollar_vol = (px["Close"] * px["Volume"]).dropna().tail(liquidity_days)
        if len(dollar_vol) < liquidity_days * 0.8:  # 2년치가 안 되면 "내내 유지" 검증 불가
            continue
        rolling = dollar_vol.rolling(cfg["SUSTAIN_WINDOW_DAYS"]).mean().dropna()
        roll_min = float(rolling.min())
        if roll_min < threshold:
            continue
        avg_2y = float(dollar_vol.mean())
        avg_recent = float(dollar_vol.tail(cfg["INTEREST_RECENT_DAYS"]).mean())
        stocks.append({
            "ticker": t,
            "name": meta.loc[t, "name"],
            "sector": meta.loc[t, "sector"],
            "price": round(float(close.iloc[-1]), 2),
            "avg_dollar_vol_b": round(avg_2y / 1e9, 2),   # 십억 달러
            "roll_min_b": round(roll_min / 1e9, 2),
            "interest_ratio": round(avg_recent / avg_2y, 2),
        })

    stocks.sort(key=lambda x: -x["avg_dollar_vol_b"])
    print(f"조건 1 통과(분기평균 거래대금 {cfg['LIQUIDITY_YEARS']}년 내내 ${cfg['MIN_DOLLAR_VOLUME_M']}M 이상): {len(stocks)}개")

    # ── 통과 종목별 10년 일봉 차트 파일 (대시보드가 행 클릭 시 fetch) ──
    shutil.rmtree("charts", ignore_errors=True)
    os.makedirs("charts")
    for s in stocks:
        close = data[s["ticker"]]["Close"].dropna()
        with open(f"charts/{s['ticker']}.json", "w") as f:
            json.dump({
                "ticker": s["ticker"],
                "dates": [str(d.date()) for d in close.index],
                "close": [float(f"{v:.6g}") for v in close.to_numpy()],
            }, f, separators=(",", ":"))
    print(f"차트 파일 {len(stocks)}개 저장 완료")

    now_utc = datetime.now(timezone.utc)
    return {
        "updated_utc": now_utc.strftime("%Y-%m-%d %H:%M UTC"),
        "updated_kst": (now_utc + timedelta(hours=9)).strftime("%Y-%m-%d %H:%M KST"),
        "config": cfg,
        "universe_count": len(members),
        "stocks": stocks,
    }


if __name__ == "__main__":
    t0 = time.time()
    result = run()
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"data.json 저장 완료 ({time.time() - t0:.0f}초)")
