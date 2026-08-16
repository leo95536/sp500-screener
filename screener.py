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

# 한국어 종목명 (국내 증권앱 통용 표기). 매핑에 없는 종목은 영문만 표시된다.
KOREAN_NAMES = {
    "NVDA": "엔비디아", "TSLA": "테슬라", "AAPL": "애플", "MU": "마이크론 테크놀로지",
    "MSFT": "마이크로소프트", "AMZN": "아마존", "META": "메타 플랫폼스", "AMD": "AMD",
    "GOOGL": "알파벳 A", "GOOG": "알파벳 C", "PLTR": "팔란티어 테크놀로지스",
    "AVGO": "브로드컴", "INTC": "인텔", "NFLX": "넷플릭스", "ORCL": "오라클",
    "LLY": "일라이 릴리", "UNH": "유나이티드헬스 그룹", "MRVL": "마벨 테크놀로지",
    "COIN": "코인베이스", "JPM": "JP모건 체이스", "APP": "앱러빈",
    "BRK-B": "버크셔 해서웨이 B", "V": "비자", "CRM": "세일즈포스", "WMT": "월마트",
    "COST": "코스트코", "AMAT": "어플라이드 머티어리얼즈", "XOM": "엑슨모빌",
    "SMCI": "슈퍼마이크로", "QCOM": "퀄컴", "BAC": "뱅크오브아메리카",
    "NOW": "서비스나우", "GEV": "GE 버노바", "LRCX": "램리서치", "BA": "보잉",
    "GS": "골드만삭스", "CSCO": "시스코", "MA": "마스터카드", "UBER": "우버",
    "JNJ": "존슨앤드존슨", "ADBE": "어도비", "CRWD": "크라우드스트라이크",
    "TXN": "텍사스 인스트루먼트", "CAT": "캐터필러", "CVX": "셰브론", "IBM": "IBM",
    "HD": "홈디포", "PANW": "팔로알토 네트웍스", "GE": "GE 에어로스페이스",
    "PG": "프록터앤갬블", "DELL": "델 테크놀로지스", "ABBV": "애브비",
    "BKNG": "부킹홀딩스", "C": "씨티그룹", "KLAC": "KLA", "INTU": "인튜이트",
    "WFC": "웰스파고", "VRT": "버티브", "MRK": "머크", "KO": "코카콜라",
    "ACN": "액센츄어", "PEP": "펩시코", "PFE": "화이자", "TMO": "서모피셔 사이언티픽",
    "ANET": "아리스타 네트웍스", "MCD": "맥도날드", "NKE": "나이키", "DIS": "월트디즈니",
    "ADI": "아날로그 디바이시스", "T": "AT&T", "CVNA": "카바나", "TMUS": "T모바일 US",
    "LIN": "린데", "ISRG": "인튜이티브 서지컬", "VZ": "버라이즌",
    "CEG": "컨스텔레이션 에너지", "MS": "모건스탠리", "AXP": "아메리칸 익스프레스",
    "ETN": "이튼", "ABT": "애보트", "SBUX": "스타벅스", "VST": "비스트라",
    "HON": "하니웰", "PYPL": "페이팔", "PM": "필립모리스", "AMGN": "암젠",
    "SCHW": "찰스슈왑", "CMCSA": "컴캐스트", "DHR": "다나허", "NEE": "넥스트에라 에너지",
    "RTX": "RTX", "COP": "코노코필립스", "SPGI": "S&P 글로벌", "UNP": "유니언 퍼시픽",
    "LMT": "록히드마틴", "TJX": "TJX", "CMG": "치폴레", "PGR": "프로그레시브",
    "TGT": "타깃", "NXPI": "NXP 반도체", "LOW": "로우스", "REGN": "리제네론",
    "SHW": "셔윈윌리엄스",
}


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
            "name_kr": KOREAN_NAMES.get(t),
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
        px = data[s["ticker"]][["Open", "High", "Low", "Close"]].dropna()
        compact = lambda col: [float(f"{v:.6g}") for v in px[col].to_numpy()]
        with open(f"charts/{s['ticker']}.json", "w") as f:
            json.dump({
                "ticker": s["ticker"],
                "dates": [str(d.date()) for d in px.index],
                "open": compact("Open"),
                "high": compact("High"),
                "low": compact("Low"),
                "close": compact("Close"),
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
