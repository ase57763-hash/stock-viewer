"""
看盤後端 v3 — 輕量版
選股結果：從 Google 試算表讀取（不在 Render 跑選股）
K線/指標：即時從 Yahoo Finance 抓
籌碼：FinMind（台股）/ Yahoo Finance（美股）
"""
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os, logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="看盤系統 v3")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

FINMIND_URL   = "https://api.finmindtrade.com/api/v4/data"
FINMIND_TOKEN = os.environ.get("FINMIND_TOKEN","")
SHEETS_URL    = os.environ.get("GOOGLE_SHEETS_URL","")  # Apps Script 網址

# ── 指標計算 ────────────────────────────────────
def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def ma(s,n):  return s.rolling(n).mean()
def macd(s):
    d=ema(s,12)-ema(s,26); e=ema(d,9); return d,e,(d-e)*2
def rsi(s,n=14):
    d=s.diff(); g=d.clip(lower=0).rolling(n).mean()
    l=(-d.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+g/l.replace(0,np.nan))
def cci(hi,lo,cl,n=20):
    tp=(hi+lo+cl)/3; m=tp.rolling(n).mean()
    md=tp.rolling(n).apply(lambda x:np.mean(np.abs(x-x.mean())))
    return (tp-m)/(0.015*md.replace(0,np.nan))
def safe(v):
    try:
        if v is None or pd.isna(v) or np.isinf(float(v)): return None
        return round(float(v),4)
    except: return None
def tolist(s): return [safe(v) for v in s]

# ── K線 API ─────────────────────────────────────
@app.get("/api/kline")
async def kline(ticker:str=Query(...), period:str=Query("6mo"), market:str=Query("TW")):
    sym=(ticker.upper()+".TW") if market=="TW" and "." not in ticker else ticker.upper()
    try:
        df=yf.download(sym,period=period,auto_adjust=True,progress=False)
        if df.empty: raise HTTPException(404,f"找不到：{sym}")
        df=df.dropna(subset=["Close"])
        cl=df["Close"].squeeze(); hi=df["High"].squeeze()
        lo=df["Low"].squeeze();   op=df["Open"].squeeze()
        vo=df["Volume"].squeeze()
        ma17=ma(cl,17); ma78=ma(cl,78)
        dif,dea,hist=macd(cl); rs=rsi(cl); cc=cci(hi,lo,cl)
        dates=[d.strftime("%m/%d") for d in df.index]
        name=sym
        try: name=yf.Ticker(sym).fast_info.display_name or sym
        except: pass
        last=safe(cl.iloc[-1]); prev=safe(cl.iloc[-2]) if len(cl)>1 else last
        chg=round(last-prev,2) if last and prev else 0
        chg_pct=round(chg/prev*100,2) if prev else 0
        m17L=safe(ma17.iloc[-1]); m78L=safe(ma78.iloc[-1])
        m17P=safe(ma17.iloc[-2]); m78P=safe(ma78.iloc[-2])
        dL=safe(dif.iloc[-1]);  eL=safe(dea.iloc[-1])
        dP=safe(dif.iloc[-2]);  eP=safe(dea.iloc[-2])
        rL=safe(rs.iloc[-1]);   rP=safe(rs.iloc[-2])
        rP2=safe(rs.iloc[-3]) if len(rs)>2 else rP
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
        return {
            "ticker":sym,"name":name,"market":market,
            "last_close":last,"prev_close":prev,"change":chg,"change_pct":chg_pct,
            "open":safe(op.iloc[-1]),"high":safe(hi.iloc[-1]),
            "low":safe(lo.iloc[-1]),"volume":int(vo.iloc[-1]) if not pd.isna(vo.iloc[-1]) else 0,
            "dates":dates,
            "ohlc":{"open":tolist(op),"high":tolist(hi),"low":tolist(lo),
                    "close":tolist(cl),"volume":[int(v) if not pd.isna(v) else 0 for v in vo]},
            "indicators":{"ma17":tolist(ma17),"ma78":tolist(ma78),
                          "dif":tolist(dif),"dea":tolist(dea),"hist":tolist(hist),
                          "rsi":tolist(rs),"cci":tolist(cc)},
            "signal":{"type":sig,"hit":hit,"cond_ma":cma,"cond_macd":cmacd,"cond_rsi":crsi,
                      "ma17":m17L,"ma78":m78L,"dif":dL,"dea":eL,"rsi":rL,"gap_pct":gap}
        }
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500,str(e))

# ── 台股籌碼 ─────────────────────────────────────
@app.get("/api/chip/tw")
async def chip_tw(ticker:str=Query(...), days:int=Query(60)):
    token=FINMIND_TOKEN
    if not token: return JSONResponse({"error":"請設定 FINMIND_TOKEN"},status_code=400)
    start=(datetime.now()-timedelta(days=days+10)).strftime("%Y-%m-%d")
    sid=ticker.replace(".TW","")
    res={"foreign":[],"invest":[],"dealer":[],"dates":[],"summary":{}}
    try:
        r=requests.get(FINMIND_URL,params={
            "dataset":"TaiwanStockInstitutionalInvestorsBuySell",
            "data_id":sid,"start_date":start,"token":token},timeout=15)
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
                "foreign_5d":int(sum(res["foreign"][-5:])),
                "invest_5d": int(sum(res["invest"][-5:])),
                "dealer_5d": int(sum(res["dealer"][-5:])),
                "foreign_10d":int(sum(res["foreign"][-10:])),
                "invest_10d": int(sum(res["invest"][-10:])),
            }
    except Exception as e: log.warning(f"chip_tw {sid}: {e}")
    try:
        r2=requests.get(FINMIND_URL,params={
            "dataset":"TaiwanStockShareholding","data_id":sid,
            "start_date":start,"token":token},timeout=15)
        d2=r2.json().get("data",[])
        if d2:
            latest=sorted(d2,key=lambda x:x.get("date",""))[-1]
            res["holding"]={
                "foreign_pct":safe(float(latest.get("ForeignInvestmentRatio",0))),
                "top10_pct":  safe(float(latest.get("top10HoldingRatio",0))),
            }
    except: pass
    return res

# ── 美股籌碼 ─────────────────────────────────────
@app.get("/api/chip/us")
async def chip_us(ticker:str=Query(...)):
    try:
        tk=yf.Ticker(ticker.upper())
        inst_list=[]
        try:
            inst=tk.institutional_holders
            if inst is not None and not inst.empty:
                for _,row in inst.head(10).iterrows():
                    inst_list.append({"name":str(row.get("Holder","—")),
                        "pct":safe(float(row.get("pctHeld",0))*100),
                        "shares":int(row.get("Shares",0))})
        except: pass
        insider_list=[]
        try:
            ins=tk.insider_transactions
            if ins is not None and not ins.empty:
                for _,row in ins.head(6).iterrows():
                    insider_list.append({"name":str(row.get("Insider","—")),
                        "shares":int(row.get("Shares",0)),
                        "type":str(row.get("Transaction","—")),
                        "date":str(row.get("Start Date","—"))})
        except: pass
        return {"institutional":inst_list,"insider":insider_list}
    except Exception as e:
        return {"institutional":[],"insider":[]}

# ── 從 Google 試算表讀取選股結果 ──────────────────
@app.get("/api/screen/cached")
async def screen_cached(market:str=Query("TW")):
    """
    從 Google Apps Script 讀取本機電腦推送的選股結果
    """
    url=SHEETS_URL
    if not url:
        return JSONResponse({"error":"請設定 GOOGLE_SHEETS_URL 環境變數","results":[]})
    try:
        r=requests.get(url,params={"action":"read","market":market},timeout=15)
        data=r.json()
        return data
    except Exception as e:
        return JSONResponse({"error":str(e),"results":[]})

@app.get("/api/health")
async def health():
    return {"status":"ok","finmind":bool(FINMIND_TOKEN),
            "sheets":bool(SHEETS_URL),"time":datetime.now().isoformat()}

# ── 靜態前端 ────────────────────────────────────
static_dir=Path(__file__).parent/"static"
if static_dir.exists():
    app.mount("/static",StaticFiles(directory=str(static_dir)),name="static")

@app.get("/")
async def root():
    idx=static_dir/"index.html"
    if idx.exists(): return FileResponse(str(idx))
    return {"status":"ok"}

if __name__=="__main__":
    import uvicorn,socket
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(("8.8.8.8",80))
        ip=s.getsockname()[0]; s.close()
    except: ip="127.0.0.1"
    print(f"\n  電腦：http://localhost:8000\n  手機：http://{ip}:8000\n")
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=False)
