import json
import re
import time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ==========================================
# 基本設定（可依需要微調）
# ==========================================
START_DATE = "2026-08-01"   # 時光機起點
USE_RESET = True            # 穿底/穿頂重數
IS_LOOSE = True             # 寬鬆模式 (允許 E 與 R 同日)
BATCH_SIZE = 50             # 每批下載股數（避免被 Yahoo 限速）

# 港交所披露易 (HKEXnews) 官方滬深港通南向全部股票頁面（含繁體中文名）
HKEXNEWS_SOUTHBOUND_URLS = [
    "https://www3.hkexnews.hk/sdw/search/mutualmarket_c.aspx?t=hk",
    "https://www3.hkexnews.hk/sdw/search/mutualmarket.aspx?t=hk",
]

# 備用港股通核心名單（萬一官方連線暫時異常時自動啟用）
FALLBACK_STOCKS = [
    ("0700.HK", "騰訊控股"), ("9988.HK", "阿里巴巴-W"), ("3690.HK", "美團-W"),
    ("1810.HK", "小米集團-W"), ("9618.HK", "京東集團-SW"), ("9888.HK", "百度集團-SW"),
    ("9999.HK", "網易-S"), ("1024.HK", "快手-W"), ("9660.HK", "地平線機器人-W"),
    ("0981.HK", "中芯國際"), ("1347.HK", "華虹半導體"), ("0005.HK", "匯豐控股"),
    ("0939.HK", "建設銀行"), ("1398.HK", "工商銀行"), ("3988.HK", "中國銀行"),
    ("1658.HK", "郵儲銀行"), ("3968.HK", "招商銀行"), ("2318.HK", "中國平安"),
    ("2628.HK", "中國人壽"), ("1299.HK", "友邦保險"), ("0388.HK", "香港交易所"),
    ("0941.HK", "中國移動"), ("0728.HK", "中國電信"), ("0762.HK", "中國聯通"),
    ("0883.HK", "中國海洋石油"), ("0857.HK", "中國石油股份"), ("0386.HK", "中國石油化工"),
    ("1088.HK", "中國神華"), ("1898.HK", "中煤能源"), ("2899.HK", "紫金礦業"),
    ("1211.HK", "比亞迪股份"), ("2015.HK", "理想汽車-W"), ("9868.HK", "小鵬汽車-W"),
    ("0175.HK", "吉利汽車"), ("2858.HK", "易鑫集團"), ("1686.HK", "新意網集團"),
    ("1133.HK", "哈爾濱電氣"), ("2269.HK", "藥明生物"), ("6160.HK", "百濟神州"),
    ("1801.HK", "信達生物"), ("9992.HK", "泡泡瑪特"), ("2020.HK", "安踏體育"),
    ("2331.HK", "李寧"), ("0669.HK", "創科實業"), ("0016.HK", "新鴻基地產"),
    ("0001.HK", "長和"), ("0002.HK", "中電控股"), ("2800.HK", "盈富基金"),
    ("3033.HK", "南方恒生科技")
]


def format_hk_ticker(code_str: str) -> str:
    digits = "".join(ch for ch in str(code_str) if ch.isdigit())
    if not digits:
        return ""
    num = int(digits)
    if num <= 0 or num > 99999:
        return ""
    return f"{num:04d}.HK"


def fetch_hk_connect_list():
    """從港交所披露易 (HKEXnews) 官方頁面直接解析全部港股通南向股票"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    zh_name_map = dict(FALLBACK_STOCKS)

    for url in HKEXNEWS_SOUTHBOUND_URLS:
        try:
            resp = requests.get(url, headers=headers, timeout=25)
            resp.raise_for_status()
            html = resp.text

            rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I)
            merged = {}
            for row in rows:
                code_m = re.search(
                    r'col-stock-code.*?<div[^>]*class="mobile-list-body"[^>]*>(.*?)</div>',
                    row, flags=re.S | re.I
                )
                name_m = re.search(
                    r'col-stock-name.*?<div[^>]*class="mobile-list-body"[^>]*>(.*?)</div>',
                    row, flags=re.S | re.I
                )
                if not code_m:
                    continue
                ticker = format_hk_ticker(re.sub(r"<[^>]+>", "", code_m.group(1)))
                if not ticker:
                    continue
                raw_name = re.sub(r"<[^>]+>", "", name_m.group(1)).strip() if name_m else ticker
                raw_name = re.sub(r"\s+", " ", raw_name)
                merged[ticker] = zh_name_map.get(ticker, raw_name or ticker)

            if len(merged) >= 100:
                for t, n in FALLBACK_STOCKS:
                    if t not in merged:
                        merged[t] = n
                result = sorted(merged.items(), key=lambda x: x[0])
                print(f"[INFO] 成功從 HKEXnews 取得 {len(result)} 隻港股通股票")
                return result, "hkexnews_southbound"
            else:
                print(f"[WARN] HKEXnews 解析筆數不足 ({len(merged)} 筆): {url}")
        except Exception as e:
            print(f"[WARN] 連線 HKEXnews 失敗: {e}")

    print("[WARN] 改用內建核心名單")
    return FALLBACK_STOCKS, "fallback_core"


def run_combo_engine(df: pd.DataFrame, combo_type: str, start_date: str, use_reset=True, is_loose=True):
    """移植自 Sky Sir Combo Alert Engine (Pine Script v6) 狀態機（含距 EP % 與新鮮度計算）"""
    n = len(df)
    if n < 5:
        return None

    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)
    dates = df.index.strftime("%Y-%m-%d").tolist()

    is_up = combo_type in ("TERE", "S陽TE", "S陽TERE")
    max_step = {"TERE": 4, "S陽TE": 3, "S陽TERE": 5, "R陰E": 2, "黑白DE": 2}[combo_type]
    fixed_sup = float(np.nanmin(lows[-20:])) if n >= 20 else float(np.nanmin(lows))

    step = 0
    is_completed = False
    ep_level = None
    reset_level = None
    track_high = None
    track_low = None
    found_s = False
    found_r = False
    s_low = None
    r_high = None
    last_event = ""
    last_event_date = ""
    started = False

    for i in range(2, n):
        d = dates[i]
        if d < start_date:
            continue

        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        prev_h, prev_l, prev_c = highs[i - 1], lows[i - 1], closes[i - 1]
        prev2_h, prev2_l = highs[i - 2], lows[i - 2]

        th = max(h, prev_c)
        tl = min(l, prev_c)
        is_bull = c > o
        is_bear = c < o
        is_swing_s = (prev_l <= prev2_l) and (prev_l <= l)
        is_swing_r = (prev_h >= prev2_h) and (prev_h >= h)
        is_s_yang = is_bull and (c > prev_h)
        is_r_yin = is_bear and (c < prev_l)

        if not started:
            started = True
            step = 0
            is_completed = False
            ep_level = None
            reset_level = None
            track_high = th
            track_low = tl
            found_s = False
            found_r = False
            s_low = None
            r_high = None

        if is_completed:
            continue

        track_high = th if track_high is None else max(track_high, th)
        track_low = tl if track_low is None else min(track_low, tl)

        if use_reset and step > 0 and reset_level is not None:
            if is_up and l < reset_level:
                step = 0
                ep_level = None
                reset_level = None
                track_high = th
                track_low = tl
                found_s = False
                s_low = None
                last_event = "Reset"
                last_event_date = d
            elif (not is_up) and h > reset_level:
                step = 0
                ep_level = None
                reset_level = None
                track_high = th
                track_low = tl
                found_r = False
                r_high = None
                last_event = "Reset"
                last_event_date = d

        prev_step = step

        if combo_type == "TERE":
            if step == 0 and is_bear:
                step = 1
                ep_level = th
                reset_level = tl
                last_event = "T"
            elif step == 1 and ep_level is not None and c > ep_level:
                step = 2
                last_event = "E"
                if is_loose and is_swing_r:
                    step = 3
                    ep_level = track_high
                    last_event = "E/R"
            elif step == 2 and is_swing_r:
                step = 3
                ep_level = track_high
                last_event = "R"
            elif step == 3 and ep_level is not None and c > ep_level:
                step = 4
                is_completed = True
                last_event = "E(完成)"

        elif combo_type == "S陽TE":
            if step == 0:
                if is_swing_s:
                    found_s = True
                    s_low = prev_l
                if found_s and is_s_yang:
                    step = 1
                    reset_level = s_low if s_low is not None else tl
                    last_event = "S陽"
            elif step == 1 and is_bear:
                step = 2
                ep_level = th
                last_event = "T"
            elif step == 2 and ep_level is not None and c > ep_level:
                step = 3
                is_completed = True
                last_event = "E(完成)"

        elif combo_type == "S陽TERE":
            if step == 0:
                if is_swing_s:
                    found_s = True
                    s_low = prev_l
                if found_s and is_s_yang:
                    step = 1
                    reset_level = s_low if s_low is not None else tl
                    last_event = "S陽"
            elif step == 1 and is_bear:
                step = 2
                ep_level = th
                last_event = "T"
            elif step == 2 and ep_level is not None and c > ep_level:
                step = 3
                last_event = "E"
                if is_loose and is_swing_r:
                    step = 4
                    ep_level = track_high
                    last_event = "E/R"
            elif step == 3 and is_swing_r:
                step = 4
                ep_level = track_high
                last_event = "R"
            elif step == 4 and ep_level is not None and c > ep_level:
                step = 5
                is_completed = True
                last_event = "E(完成)"

        elif combo_type == "R陰E":
            if step == 0:
                if is_swing_r:
                    found_r = True
                    r_high = prev_h
                if found_r and is_r_yin:
                    step = 1
                    reset_level = r_high if r_high is not None else th
                    ep_level = tl
                    last_event = "R陰"
            elif step == 1 and ep_level is not None and c < ep_level:
                step = 2
                is_completed = True
                last_event = "E(完成)"

        elif combo_type == "黑白DE":
            if step == 0 and is_bear and c < fixed_sup:
                step = 1
                ep_level = tl
                reset_level = th
                last_event = "D"
            elif step == 1 and ep_level is not None and c < ep_level:
                step = 2
                is_completed = True
                last_event = "E(完成)"

        if step != prev_step:
            last_event_date = d

    # 新增：計算現價距離 EP 位的百分比 (%)
    latest_close = float(closes[-1])
    dist_to_ep_pct = None
    if ep_level is not None and ep_level > 0:
        dist_to_ep_pct = round(((latest_close - ep_level) / ep_level) * 100.0, 2)

    # 新增：計算最近一次訊號觸發距離最新交易日相隔幾個交易日
    days_since_event = None
    if last_event_date and last_event_date in dates:
        event_idx = dates.index(last_event_date)
        days_since_event = (n - 1) - event_idx

    return {
        "step": step,
        "max_step": max_step,
        "is_up": is_up,
        "completed": is_completed,
        "ep_level": round(ep_level, 3) if ep_level is not None else None,
        "reset_level": round(reset_level, 3) if reset_level is not None else None,
        "dist_to_ep_pct": dist_to_ep_pct,
        "days_since_event": days_since_event,
        "last_event": last_event,
        "last_event_date": last_event_date
    }


def extract_single_df(batch_df: pd.DataFrame, ticker: str, batch_tickers: list):
    if batch_df is None or batch_df.empty:
        return None
    if isinstance(batch_df.columns, pd.MultiIndex):
        if ticker in batch_df.columns.get_level_values(0):
            sub = batch_df[ticker].copy()
        else:
            return None
    else:
        if len(batch_tickers) == 1:
            sub = batch_df.copy()
        else:
            return None
    sub = sub.dropna(subset=["Open", "High", "Low", "Close"])
    return sub if len(sub) >= 10 else None


def main():
    stocks, source_name = fetch_hk_connect_list()
    combos_list = ["TERE", "S陽TE", "S陽TERE", "R陰E", "黑白DE"]
    results = []

    total = len(stocks)
    for start_idx in range(0, total, BATCH_SIZE):
        batch = stocks[start_idx:start_idx + BATCH_SIZE]
        tickers = [t for t, _ in batch]
        name_map = dict(batch)
        print(f"[INFO] 正在下載第 {start_idx + 1} - {min(start_idx + BATCH_SIZE, total)} / {total} 隻...")

        try:
            data = yf.download(
                tickers=tickers,
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False
            )
        except Exception as e:
            print(f"[WARN] 批次下載失敗: {e}")
            continue

        for t in tickers:
            df = extract_single_df(data, t, tickers)
            if df is None or df.empty:
                continue

            last_close = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else last_close
            chg_pct = ((last_close - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
            last_vol = float(df["Volume"].iloc[-1]) if "Volume" in df.columns and not pd.isna(df["Volume"].iloc[-1]) else 0.0
            turnover_hkd = last_close * last_vol
            last_date = df.index[-1].strftime("%Y-%m-%d")
            sparkline = [round(float(x), 3) for x in df["Close"].tail(20).tolist()]

            combo_states = {}
            for c_type in combos_list:
                res = run_combo_engine(df, c_type, START_DATE, USE_RESET, IS_LOOSE)
                if res:
                    combo_states[c_type] = res

            results.append({
                "ticker": t,
                "name": name_map.get(t, t),
                "close": round(last_close, 3),
                "change_pct": round(chg_pct, 2),
                "volume": int(last_vol),
                "turnover_m": round(turnover_hkd / 1e6, 2),
                "last_date": last_date,
                "sparkline": sparkline,
                "combos": combo_states
            })

        time.sleep(1.0)

    hkt = timezone(timedelta(hours=8))
    output = {
        "updated_at": datetime.now(hkt).strftime("%Y-%m-%d %H:%M HKT"),
        "start_date": START_DATE,
        "source": source_name,
        "total_scanned": len(results),
        "stocks": results
    }

    with open("scan_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, separators=(",", ":"))

    print(f"[DONE] 掃描完成！共寫入 {len(results)} 隻港股通股票至 scan_results.json")


if __name__ == "__main__":
    main()
