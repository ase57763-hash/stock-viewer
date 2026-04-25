"""
看盤後端 v4 — 永豐 Shioaji API 版
台股/美股 K線：永豐 Shioaji（取代 yfinance）
台股籌碼：FinMind
選股結果：Google 試算表
"""
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import requests
import os, logging
from datetime import datetime, timedelta, date
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="看盤系統 v4 Shioaji")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

FINMIND_URL   = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")
SHEETS_URL    = os.environ.get("GOOGLE_SHEETS_URL", "")
SHIOAJI_KEY   = os.environ.get("SHIOAJI_API_KEY", "")
SHIOAJI_SEC   = os.environ.get("SHIOAJI_SECRET_KEY", "")

# ── Shioaji 連線（單例，避免重複登入）──────────────
_sj_api = None

def get_sj():
    global _sj_api
    if _sj_api is not None:
        return _sj_api
    if not SHIOAJI_KEY or not SHIOAJI_SEC:
        return None
    try:
        import shioaji as sj
        api = sj.Shioaji()
        api.login(api_key=SHIOAJI_KEY, secret_key=SHIOAJI_SEC,
                  receive_window=60000)
        _sj_api = api
        log.info("Shioaji 登入成功")
        return api
    except Exception as e:
        log.error(f"Shioaji 登入失敗：{e}")
        return None

# ── 指標計算 ────────────────────────────────────
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def ma(s, n):  return s.rolling(n).mean()
def macd_calc(s):
    d = ema(s,12)-ema(s,26); e = ema(d,9); return d, e, (d-e)*2
def rsi_calc(s, n=14):
    d = s.diff(); g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100/(1+g/l.replace(0, np.nan))
def cci_calc(hi, lo, cl, n=20):
    tp = (hi+lo+cl)/3; m = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x-x.mean())))
    return (tp-m)/(0.015*md.replace(0, np.nan))

def safe(v):
    try:
        if v is None or pd.isna(v) or np.isinf(float(v)): return None
        return round(float(v), 4)
    except: return None

def tolist(s): return [safe(v) for v in s]

def build_signal(ma17, ma78, dif_s, dea_s, rsi_s, cl):
    """計算訊號"""
    m17L=safe(ma17.iloc[-1]); m78L=safe(ma78.iloc[-1])
    m17P=safe(ma17.iloc[-2]); m78P=safe(ma78.iloc[-2])
    dL=safe(dif_s.iloc[-1]);  eL=safe(dea_s.iloc[-1])
    dP=safe(dif_s.iloc[-2]);  eP=safe(dea_s.iloc[-2])
    rL=safe(rsi_s.iloc[-1]);  rP=safe(rsi_s.iloc[-2])
    rP2=safe(rsi_s.iloc[-3]) if len(rsi_s)>2 else rP
    cma  =bool(m17L and m78L and m17P and m78P and m17P<=m78P and m17L>m78L)
    cmacd=bool(dL and eL and dP and eP and dP<=eP and dL>eL)
    crsi =bool(rL and rP and rP2 and rL>rP and rP<rP2 and 30<rL<80)
    hit=sum([cma,cmacd,crsi])
    gap=round((m17L-m78L)/m78L*100,2) if m17L and m78L else 0
    trend=(m17L-m17P) if m17L and m17P else 0
    if   hit==3: sig="條件三中三"
    elif hit==2: sig="條件三中二"
    elif gap<0 and gap>-1.5 and trend>0: sig="即將黃金交叉"
    elif gap>0 and gap<1.5  and trend<0: sig="即將死亡交叉"
    else: sig="觀察中"
    return {"type":sig,"hit":hit,"cond_ma":cma,"cond_macd":cmacd,"cond_rsi":crsi,
            "ma17":m17L,"ma78":m78L,"dif":dL,"dea":eL,"rsi":rL,"gap_pct":gap}

def period_to_dates(period: str):
    """把 3mo/6mo/1y/2y 轉成 start/end date"""
    end = date.today()
    mapping = {"3mo":90,"6mo":180,"1y":365,"2y":730}
    days = mapping.get(period, 180)
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

# ── 永豐 台股 K線 ────────────────────────────────
def fetch_tw_kline(ticker: str, period: str) -> pd.DataFrame:
    """用 Shioaji 抓台股 K線，回傳 DataFrame"""
    api = get_sj()
    if api is None:
        raise Exception("Shioaji 未連線，請確認 API Key 設定")

    import shioaji as sj
    start, end = period_to_dates(period)

    contract = api.Contracts.Stocks.TSE.get(ticker) or \
               api.Contracts.Stocks.OTC.get(ticker)
    if contract is None:
        raise Exception(f"找不到台股代碼：{ticker}")

    kbars = api.kbars(contract=contract,
                      start=start, end=end,
                      timeout=30000)
    df = pd.DataFrame({**kbars})
    if df.empty:
        raise Exception(f"無法取得 {ticker} K線資料")

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").rename(columns={
        "Open":"Open","High":"High","Low":"Low",
        "Close":"Close","Volume":"Volume"
    })
    df.set_index("ts", inplace=True)
    return df

# ── 永豐 美股 K線（複委託）──────────────────────
def fetch_us_kline(ticker: str, period: str) -> pd.DataFrame:
    """用 Shioaji 抓美股複委託 K線"""
    api = get_sj()
    if api is None:
        raise Exception("Shioaji 未連線，請確認 API Key 設定")

    start, end = period_to_dates(period)

    # 美股複委託合約
    contract = None
    try:
        contract = api.Contracts.Stocks.NYSE.get(ticker) or \
                   api.Contracts.Stocks.NASDAQ.get(ticker) or \
                   api.Contracts.Stocks.AMEX.get(ticker)
    except: pass

    if contract is None:
        # 嘗試直接用代碼搜尋
        try:
            results = api.search_contracts(ticker)
            for r in results:
                if hasattr(r, "exchange") and r.exchange in ("NYSE","NASDAQ","AMEX"):
                    contract = r; break
        except: pass

    if contract is None:
        raise Exception(f"找不到美股代碼：{ticker}（請確認複委託已開通）")

    kbars = api.kbars(contract=contract,
                      start=start, end=end,
                      timeout=30000)
    df = pd.DataFrame({**kbars})
    if df.empty:
        raise Exception(f"無法取得 {ticker} K線資料")

    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").rename(columns={
        "Open":"Open","High":"High","Low":"Low",
        "Close":"Close","Volume":"Volume"
    })
    df.set_index("ts", inplace=True)
    return df

# ── K線 API ─────────────────────────────────────
@app.get("/api/kline")
async def kline(ticker:str=Query(...), period:str=Query("6mo"), market:str=Query("TW")):
    sym = ticker.upper().replace(".TW","")
    log.info(f"kline {sym} {period} {market}")
    try:
        if market == "TW":
            df = fetch_tw_kline(sym, period)
        else:
            df = fetch_us_kline(sym, period)

        if df.empty: raise HTTPException(404, f"找不到：{sym}")
        df = df.dropna(subset=["Close"])

        cl = df["Close"].squeeze(); hi = df["High"].squeeze()
        lo = df["Low"].squeeze();   op = df["Open"].squeeze()
        vo = df["Volume"].squeeze()

        ma17 = ma(cl,17); ma78 = ma(cl,78)
        dif_s, dea_s, hist_s = macd_calc(cl)
        rs = rsi_calc(cl); cc = cci_calc(hi, lo, cl)

        dates = [d.strftime("%m/%d") for d in df.index]
        last  = safe(cl.iloc[-1]); prev = safe(cl.iloc[-2]) if len(cl)>1 else last
        chg   = round(last-prev, 2) if last and prev else 0
        chg_pct = round(chg/prev*100, 2) if prev else 0

        sig = build_signal(ma17, ma78, dif_s, dea_s, rs, cl)

        return {
            "ticker": sym, "name": sym, "market": market,
            "last_close": last, "prev_close": prev,
            "change": chg, "change_pct": chg_pct,
            "open": safe(op.iloc[-1]), "high": safe(hi.iloc[-1]),
            "low":  safe(lo.iloc[-1]),
            "volume": int(vo.iloc[-1]) if not pd.isna(vo.iloc[-1]) else 0,
            "dates": dates,
            "ohlc": {
                "open":   tolist(op), "high": tolist(hi),
                "low":    tolist(lo), "close": tolist(cl),
                "volume": [int(v) if not pd.isna(v) else 0 for v in vo]
            },
            "indicators": {
                "ma17": tolist(ma17), "ma78": tolist(ma78),
                "dif":  tolist(dif_s), "dea": tolist(dea_s),
                "hist": tolist(hist_s), "rsi": tolist(rs),
                "cci":  tolist(cc)
            },
            "signal": sig
        }
    except HTTPException: raise
    except Exception as e:
        log.error(f"kline error {sym}: {e}")
        raise HTTPException(500, str(e))

# ── 台股籌碼（FinMind）──────────────────────────
@app.get("/api/chip/tw")
async def chip_tw(ticker:str=Query(...), days:int=Query(60)):
    if not FINMIND_TOKEN:
        return JSONResponse({"error":"請設定 FINMIND_TOKEN"},status_code=400)
    start=(datetime.now()-timedelta(days=days+10)).strftime("%Y-%m-%d")
    sid=ticker.replace(".TW","")
    res={"foreign":[],"invest":[],"dealer":[],"dates":[],"summary":{}}
    try:
        r=requests.get(FINMIND_URL,params={
            "dataset":"TaiwanStockInstitutionalInvestorsBuySell",
            "data_id":sid,"start_date":start,"token":FINMIND_TOKEN},timeout=15)
        data=r.json().get("data",[])
        if data:
            df=pd.DataFrame(data).sort_values("date")
            dates=sorted(df["date"].unique().tolist())[-days:]
            dates_fmt=[d[5:].replace("-","/") for d in dates]
            for name,key in [("外資","foreign"),("投信","invest"),("自營商","dealer")]:
                sub=df[df["name"]==name] if "name" in df.columns else pd.DataFrame()
                if not sub.empty:
                    b=pd.to_numeric(sub.get("buy",pd.Series(dtype=float)),errors="coerce").fillna(0)
                    s=pd.to_numeric(sub.get("sell",pd.Series(dtype=float)),errors="coerce").fillna(0)
                    res[key]=(b-s).tail(days).tolist()
                else: res[key]=[0]*len(dates_fmt)
            res["dates"]=dates_fmt
            res["summary"]={
                "foreign_5d": int(sum(res["foreign"][-5:])),
                "invest_5d":  int(sum(res["invest"][-5:])),
                "dealer_5d":  int(sum(res["dealer"][-5:])),
                "foreign_10d":int(sum(res["foreign"][-10:])),
                "invest_10d": int(sum(res["invest"][-10:])),
            }
    except Exception as e: log.warning(f"chip_tw {sid}: {e}")
    try:
        r2=requests.get(FINMIND_URL,params={
            "dataset":"TaiwanStockShareholding","data_id":sid,
            "start_date":start,"token":FINMIND_TOKEN},timeout=15)
        d2=r2.json().get("data",[])
        if d2:
            latest=sorted(d2,key=lambda x:x.get("date",""))[-1]
            res["holding"]={
                "foreign_pct":safe(float(latest.get("ForeignInvestmentRatio",0))),
                "top10_pct":  safe(float(latest.get("top10HoldingRatio",0))),
            }
    except: pass
    return res

# ── 美股籌碼（Shioaji 快照）────────────────────
@app.get("/api/chip/us")
async def chip_us(ticker:str=Query(...)):
    """美股用 Shioaji 取得即時快照資訊"""
    api = get_sj()
    if api is None:
        return {"institutional":[],"insider":[],"snapshot":None}
    try:
        contract = None
        try:
            contract = api.Contracts.Stocks.NASDAQ.get(ticker.upper()) or \
                       api.Contracts.Stocks.NYSE.get(ticker.upper())
        except: pass
        if contract is None:
            return {"institutional":[],"insider":[],"snapshot":None}
        snapshots = api.snapshots([contract])
        if snapshots:
            s = snapshots[0]
            snapshot = {
                "close":      safe(s.close),
                "open":       safe(s.open),
                "high":       safe(s.high),
                "low":        safe(s.low),
                "volume":     int(s.volume) if s.volume else 0,
                "change_pct": safe(s.change_price/s.close*100) if s.close else None,
            }
            return {"institutional":[],"insider":[],"snapshot":snapshot}
    except Exception as e:
        log.warning(f"chip_us {ticker}: {e}")
    return {"institutional":[],"insider":[],"snapshot":None}

# ── 從 Google 試算表讀取選股結果 ────────────────
@app.get("/api/screen/cached")
async def screen_cached(market:str=Query("TW")):
    if not SHEETS_URL:
        return JSONResponse({"error":"請設定 GOOGLE_SHEETS_URL","results":[]})
    try:
        r=requests.get(SHEETS_URL,
                       params={"action":"read","market":market},
                       timeout=15)
        return r.json()
    except Exception as e:
        return JSONResponse({"error":str(e),"results":[]})

@app.get("/api/sectors")
async def sectors(market:str=Query("TW")):
    TW = ["半導體","AI伺服器","電動車","金融","生技醫療","傳產原物料","被動元件"]
    US = ["科技","AI半導體","電動車能源","生技醫療","光通訊","能源","軍工","零售"]
    return {"sectors": TW if market=="TW" else US}

@app.get("/api/health")
async def health():
    sj_ok = get_sj() is not None
    return {"status":"ok","shioaji":sj_ok,
            "finmind":bool(FINMIND_TOKEN),
            "sheets":bool(SHEETS_URL),
            "time":datetime.now().isoformat()}

# ── 靜態前端 ────────────────────────────────────
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    idx = static_dir / "index.html"
    if idx.exists(): return FileResponse(str(idx))
    return {"status":"ok"}

if __name__ == "__main__":
    import uvicorn, socket
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80)); ip=s.getsockname()[0]; s.close()
    except: ip="127.0.0.1"
    print(f"\n  電腦：http://localhost:8000\n  手機：http://{ip}:8000\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
