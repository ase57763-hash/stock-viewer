"""
看盤系統後端 — Render.com 部署版
"""
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os, logging, time
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="看盤系統", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

FINMIND_URL   = "https://api.finmindtrade.com/api/v4/data"
# 環境變數取 Token（在 Render 設定）
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN", "")

# ── 指標計算 ────────────────────────────────────
def calc_ema(s, n): return s.ewm(span=n, adjust=False).mean()
def calc_ma(s, n):  return s.rolling(n).mean()
def calc_macd(s):
    dif = calc_ema(s,12) - calc_ema(s,26); dea = calc_ema(dif,9)
    return dif, dea, (dif - dea) * 2
def calc_rsi(s, n=14):
    d = s.diff(); g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))
def calc_cci(hi, lo, cl, n=20):
    tp = (hi+lo+cl)/3; ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x-x.mean())))
    return (tp-ma)/(0.015*md.replace(0,np.nan))

def safe(v):
    if v is None: return None
    try:
        if pd.isna(v) or np.isinf(float(v)): return None
        return round(float(v), 4)
    except: return None

def to_list(s): return [safe(v) for v in s]

# ── K 線 + 技術指標 ─────────────────────────────
@app.get("/api/kline")
async def kline(ticker: str = Query(...), period: str = Query("6mo"), market: str = Query("TW")):
    sym = (ticker.upper()+".TW") if market=="TW" and "." not in ticker else ticker.upper()
    log.info(f"kline {sym} {period}")
    try:
        df = yf.download(sym, period=period, auto_adjust=True, progress=False)
        if df.empty: raise HTTPException(404, f"找不到：{sym}")
        df = df.dropna(subset=["Close"])
        cl = df["Close"].squeeze(); hi = df["High"].squeeze()
        lo = df["Low"].squeeze();   op = df["Open"].squeeze()
        vo = df["Volume"].squeeze()

        ma17 = calc_ma(cl,17); ma78 = calc_ma(cl,78)
        dif, dea, hist = calc_macd(cl)
        rsi = calc_rsi(cl); cci = calc_cci(hi,lo,cl,20)

        dates = [d.strftime("%m/%d") for d in df.index]

        # 取股票名稱
        name = sym
        try:
            info = yf.Ticker(sym).fast_info
            name = getattr(info,"display_name",None) or sym
        except: pass

        last = safe(cl.iloc[-1]); prev = safe(cl.iloc[-2]) if len(cl)>1 else last
        chg = round(last-prev, 2) if last and prev else 0
        chg_pct = round(chg/prev*100, 2) if prev else 0

        # 訊號
        m17L=safe(ma17.iloc[-1]); m78L=safe(ma78.iloc[-1])
        m17P=safe(ma17.iloc[-2]); m78P=safe(ma78.iloc[-2])
        difL=safe(dif.iloc[-1]);  deaL=safe(dea.iloc[-1])
        difP=safe(dif.iloc[-2]);  deaP=safe(dea.iloc[-2])
        rL=safe(rsi.iloc[-1]);    rP=safe(rsi.iloc[-2])
        rP2=safe(rsi.iloc[-3]) if len(rsi)>2 else rP

        cma   = bool(m17L and m78L and m17P and m78P and m17P<=m78P and m17L>m78L)
        cmacd = bool(difL and deaL and difP and deaP and difP<=deaP and difL>deaL)
        crsi  = bool(rL and rP and rP2 and rL>rP and rP<rP2 and 30<rL<80)
        hit   = sum([cma,cmacd,crsi])
        gap   = round((m17L-m78L)/m78L*100, 2) if m17L and m78L else 0
        trend = (m17L-m17P) if m17L and m17P else 0

        if   hit==3: sig="條件三中三"
        elif hit==2: sig="條件三中二"
        elif gap<0 and gap>-1.5 and trend>0: sig="即將黃金交叉"
        elif gap>0 and gap<1.5  and trend<0: sig="即將死亡交叉"
        else: sig="觀察中"

        return {
            "ticker":sym,"name":name,"market":market,
            "last_close":last,"prev_close":prev,"change":chg,"change_pct":chg_pct,
            "open":safe(op.iloc[-1]),"high":safe(hi.iloc[-1]),
            "low":safe(lo.iloc[-1]),"volume":int(vo.iloc[-1]) if not pd.isna(vo.iloc[-1]) else 0,
            "dates":dates,
            "ohlc":{"open":to_list(op),"high":to_list(hi),"low":to_list(lo),
                    "close":to_list(cl),"volume":[int(v) if not pd.isna(v) else 0 for v in vo]},
            "indicators":{"ma17":to_list(ma17),"ma78":to_list(ma78),
                          "dif":to_list(dif),"dea":to_list(dea),"hist":to_list(hist),
                          "rsi":to_list(rsi),"cci":to_list(cci)},
            "signal":{"type":sig,"hit":hit,"cond_ma":cma,"cond_macd":cmacd,"cond_rsi":crsi,
                      "ma17":m17L,"ma78":m78L,"dif":difL,"dea":deaL,
                      "rsi":rL,"gap_pct":gap}
        }
    except HTTPException: raise
    except Exception as e:
        log.error(f"kline error {sym}: {e}")
        raise HTTPException(500, str(e))

# ── 台股籌碼（FinMind 真實）──────────────────────
@app.get("/api/chip/tw")
async def chip_tw(ticker: str = Query(...), days: int = Query(60)):
    token = FINMIND_TOKEN
    if not token:
        return JSONResponse({"error":"請設定 FINMIND_TOKEN 環境變數"}, status_code=400)
    start = (datetime.now()-timedelta(days=days+10)).strftime("%Y-%m-%d")
    sid   = ticker.replace(".TW","")
    res   = {"foreign":[],"invest":[],"dealer":[],"dates":[],"summary":{}}
    try:
        r = requests.get(FINMIND_URL, params={
            "dataset":"TaiwanStockInstitutionalInvestorsBuySell",
            "data_id":sid,"start_date":start,"token":token}, timeout=15)
        data = r.json().get("data",[])
        if data:
            df = pd.DataFrame(data).sort_values("date")
            df["date"] = pd.to_datetime(df["date"])
            dates = sorted(df["date"].dt.strftime("%m/%d").unique().tolist())[-days:]
            for name, key in [("外資","foreign"),("投信","invest"),("自營商","dealer")]:
                sub = df[df["name"]==name] if "name" in df.columns else pd.DataFrame()
                if not sub.empty:
                    sub = sub.tail(days)
                    buy  = pd.to_numeric(sub.get("buy",  pd.Series(dtype=float)), errors="coerce").fillna(0)
                    sell = pd.to_numeric(sub.get("sell", pd.Series(dtype=float)), errors="coerce").fillna(0)
                    res[key] = (buy-sell).tolist()
                else:
                    res[key] = [0]*len(dates)
            res["dates"] = dates
            res["summary"] = {
                "foreign_5d":  int(sum(res["foreign"][-5:])),
                "invest_5d":   int(sum(res["invest"][-5:])),
                "dealer_5d":   int(sum(res["dealer"][-5:])),
                "foreign_10d": int(sum(res["foreign"][-10:])),
                "invest_10d":  int(sum(res["invest"][-10:])),
            }
    except Exception as e:
        log.warning(f"FinMind chip {sid}: {e}")

    # 持股比例
    try:
        r2 = requests.get(FINMIND_URL, params={
            "dataset":"TaiwanStockShareholding","data_id":sid,
            "start_date":start,"token":token}, timeout=15)
        d2 = r2.json().get("data",[])
        if d2:
            latest = sorted(d2, key=lambda x: x.get("date",""))[-1]
            res["holding"] = {
                "foreign_pct": safe(float(latest.get("ForeignInvestmentRatio",0))),
                "top10_pct":   safe(float(latest.get("top10HoldingRatio",0))),
            }
    except: pass
    return res

# ── 美股籌碼（Yahoo Finance）────────────────────
@app.get("/api/chip/us")
async def chip_us(ticker: str = Query(...)):
    try:
        tk = yf.Ticker(ticker.upper())
        inst_list = []
        try:
            inst = tk.institutional_holders
            if inst is not None and not inst.empty:
                for _, row in inst.head(10).iterrows():
                    inst_list.append({
                        "name":   str(row.get("Holder","—")),
                        "pct":    safe(float(row.get("pctHeld",0))*100),
                        "shares": int(row.get("Shares",0)),
                    })
        except: pass
        insider_list = []
        try:
            ins = tk.insider_transactions
            if ins is not None and not ins.empty:
                for _, row in ins.head(6).iterrows():
                    insider_list.append({
                        "name":  str(row.get("Insider","—")),
                        "shares":int(row.get("Shares",0)),
                        "type":  str(row.get("Transaction","—")),
                        "date":  str(row.get("Start Date","—")),
                    })
        except: pass
        return {"institutional":inst_list,"insider":insider_list,"summary":{}}
    except Exception as e:
        log.warning(f"chip_us {ticker}: {e}")
        return {"institutional":[],"insider":[],"summary":{}}

# ── 即時選股 ────────────────────────────────────
SECTORS_TW = {
    "半導體":  ["2330","2454","2408","3711","2379","2303","6770"],
    "AI伺服器":["2317","2324","3034","6669","3231","4938","2356"],
    "電動車":  ["2207","1519","1537","2308","6213","3005"],
    "金融":    ["2882","2881","2891","2886","2884","2892"],
    "生技醫療":["4711","6548","1788","4746","6196"],
    "傳產":    ["1301","1303","1326","2002","1402","9904"],
    "被動元件":["2327","2492","2351","3037","2449"],
}
SECTORS_US = {
    "科技":    ["AAPL","MSFT","META","GOOGL","AMZN","ORCL","ADBE"],
    "AI半導體":["NVDA","AMD","AVGO","QCOM","ARM","INTC","MRVL"],
    "電動車":  ["TSLA","RIVN","ENPH","NEE","FSLR","NIO"],
    "生技":    ["JNJ","PFE","MRNA","ABBV","LLY","AMGN"],
    "光通訊":  ["COHR","LITE","VIAV","CIEN","INFN"],
    "能源":    ["XOM","CVX","COP","SLB","EOG","OXY"],
    "軍工":    ["LMT","RTX","NOC","GD","BA","LHX"],
    "零售":    ["WMT","COST","TGT","HD","BABA","MELI"],
}

@app.get("/api/screen")
async def screen(market: str = Query("TW"), sectors: str = Query("")):
    sec_list = [s.strip() for s in sectors.split(",") if s.strip()]
    src = SECTORS_TW if market=="TW" else SECTORS_US
    if not sec_list: sec_list = list(src.keys())
    tickers = {}
    sfx = ".TW" if market=="TW" else ""
    for sec in sec_list:
        for t in src.get(sec,[]): tickers[t+sfx] = sec

    results = []
    end = datetime.now(); start = end-timedelta(days=200)
    for ticker, sector in list(tickers.items())[:30]:
        try:
            df = yf.download(ticker,start=start,end=end,progress=False,auto_adjust=True)
            if df.empty or len(df)<85: continue
            c = df["Close"].squeeze()
            m17=calc_ma(c,17); m78=calc_ma(c,78)
            dif_s,dea_s,_ = calc_macd(c); rsi_s=calc_rsi(c)
            jn = pd.concat([m17,m78,dif_s,dea_s,rsi_s],axis=1).dropna()
            if len(jn)<3: continue
            L,P,P2 = jn.iloc[-1],jn.iloc[-2],jn.iloc[-3]
            m17L,m78L=float(L.iloc[0]),float(L.iloc[1])
            m17P,m78P=float(P.iloc[0]),float(P.iloc[1])
            difL,deaL=float(L.iloc[2]),float(L.iloc[3])
            difP,deaP=float(P.iloc[2]),float(P.iloc[3])
            rL,rP,rP2f=float(L.iloc[4]),float(P.iloc[4]),float(P2.iloc[4])
            cma  =(m17P<=m78P)and(m17L>m78L)
            cmacd=(difP<=deaP)and(difL>deaL)
            crsi =(rL>rP)and(rP<rP2f)and(30<rL<80)
            hit=sum([cma,cmacd,crsi]); gap=(m17L-m78L)/m78L*100; trend=m17L-m17P
            if   hit==3: sig="條件三中三"
            elif hit==2: sig="條件三中二"
            elif gap<0 and gap>-1.5 and trend>0: sig="即將黃金交叉"
            elif gap>0 and gap<1.5  and trend<0: sig="即將死亡交叉"
            else: continue
            results.append({"ticker":ticker,"stock_id":ticker.replace(".TW",""),
                "sector":sector,"signal":sig,"close":round(float(c.iloc[-1]),2),
                "rsi":round(rL,1),"ma17":round(m17L,2),"ma78":round(m78L,2),
                "gap_pct":round(gap,2),"cond_ma":cma,"cond_macd":cmacd,"cond_rsi":crsi})
            time.sleep(0.15)
        except: continue
    sig_order={"條件三中三":1,"條件三中二":2,"即將黃金交叉":3,"即將死亡交叉":4}
    results.sort(key=lambda x:(sig_order.get(x["signal"],9),-x["rsi"]))
    return {"market":market,"count":len(results),"results":results,
            "screened_at":datetime.now().isoformat()}

@app.get("/api/sectors")
async def sectors(market: str = Query("TW")):
    src = SECTORS_TW if market=="TW" else SECTORS_US
    return {"sectors":list(src.keys())}

@app.get("/api/health")
async def health():
    return {"status":"ok","finmind":bool(FINMIND_TOKEN),
            "time":datetime.now().isoformat()}

# ── 靜態前端 ────────────────────────────────────
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    idx = static_dir / "index.html"
    if idx.exists(): return FileResponse(str(idx))
    return {"status":"ok","message":"看盤 API running"}

if __name__=="__main__":
    import uvicorn, socket
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(("8.8.8.8",80))
        ip=s.getsockname()[0]; s.close()
    except: ip="127.0.0.1"
    print(f"\n  電腦：http://localhost:8000\n  手機：http://{ip}:8000\n")
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=False)
