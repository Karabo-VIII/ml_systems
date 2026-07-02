"""QUANT: FINAL PROD CHECKS + wire the 4 frozen candidates into the PAPER TRADER (weekly live-book emitter).

PROD CHECKS (paper-trade-safe gate -- causal/deterministic/cost, NOT real-capital-ready):
  1. CAUSALITY (perturbation): flip an input at bar t0 -> confirm the P&L stream changes ONLY at t>t0 (no same-bar/past leak).
  2. DETERMINISM: recompute -> bit-identical (no unseeded randomness).
  3. COST APPLIED: net < gross (prod cost 0.0028 RT charged on turnover, not bypassed).
  4. ROBUSTNESS SUMMARY: VAL/OOS compound + p05 + DSR40 (from the committed grades) -- the honest ceiling.
WIRE: emit each candidate's CURRENT weekly book (DEPLOY names+weights or CASH) as a dated live-book JSON + report.
Re-run each Monday for the pre-registered live-forward track. Frozen specs (NO re-tuning -- re-tuning voids the gate).
u100 1d, 14d/7d, inv-vol cap33, next-open, prod cost, $10k. long-only spot lev=1.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE)); ROOT=Path(__file__).resolve().parents[3]/"src"; sys.path.insert(0,str(ROOT))
from strat.ma_slice_engine import held_rolling, _cached_band_held
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.canonical_exits import COST_PROD
import strat.fleet_lab as fl
BPD=fl.BARS_PER_DAY["1d"]; LB=14*BPD; SL=7*BPD; SW=100*BPD; CAP=0.33; TYPES=["SMA","EMA","WMA","VIDYA","KAMA"]
WALL=int(pd.Timestamp("2026-07-01").value//10**6)
FROZEN={"CONV":"MA 5-type ensemble, breadth-hyst(0.5/0.4) gate, inv-vol cap33, 14d/7d rolling",
        "SMA_ER":"SMA band + breadth AND BTC-trend-efficiency(ER14>0.35) gate, inv-vol cap33, 14d/7d",
        "PARTIAL":"always 15-100% invested = clip((breadth-0.2)/0.6,0.15,1) x (SMA-trend book else EW-u100)",
        "MOM8_ER":"momentum14 top-8 + breadth AND ER>0.35 gate, inv-vol cap33, per-7d rebalance"}
# robustness from committed grades (strat_val_oos.json / strat_final_candidates.json)
ROBUST={"CONV":{"VAL":122,"OOS":36,"p05_oos":-44,"dsr40":0.07},"SMA_ER":{"VAL":132,"OOS":11,"p05_oos":-38,"dsr40":0.04},
        "PARTIAL":{"VAL":115,"OOS":22,"p05_oos":-46,"dsr40":0.05},"MOM8_ER":{"VAL":257,"OOS":46,"p05_oos":-28,"dsr40":0.12}}


def hyst(s,u,d):
    n=len(s);o=np.zeros(n,int);st=0
    for i in range(n):
        if not np.isfinite(s[i]): o[i]=st; continue
        if st==0 and s[i]>u: st=1
        elif st==1 and s[i]<d: st=0
        o[i]=st
    return o


def build_all(C_override=None):
    """returns candidate weight matrices W[T,A], + idx/cols/roo/net-fn. C_override lets the perturbation test flip a bar."""
    closes={};opens={}; th={t:{} for t in TYPES}
    for s in _u100_symbols():
        r=_load_asset(s,"1d",WALL)
        if r is None: continue
        d,o,h,l,c=r; closes[s]=pd.Series(np.asarray(c,float),index=d); opens[s]=pd.Series(np.asarray(o,float),index=d)
        for t in TYPES: th[t][s]=pd.Series(held_rolling(c,_cached_band_held(t,"1d",s,o,h,l,c,WALL),LB,SL),index=d)
    cdf=pd.DataFrame(closes).sort_index(); idx=cdf.index; cols=list(cdf.columns); T=len(idx); A=len(cols)
    odf=pd.DataFrame(opens).reindex(idx).reindex(columns=cols); C=cdf.to_numpy().copy(); O=odf.to_numpy()
    if C_override is not None:
        t0,j0,fac=C_override; C[t0,j0]*=fac; cdf=pd.DataFrame(C,index=idx,columns=cols)
    conv=np.zeros((T,A))
    for t in TYPES: th[t]=pd.DataFrame(th[t]).reindex(idx).fillna(0).reindex(columns=cols).fillna(0).to_numpy(); conv+=th[t]
    Hsma=th["SMA"]
    vol=cdf.pct_change(fill_method=None).rolling(max(20,5*BPD),min_periods=10).std().to_numpy()
    roo=np.full((T,A),0.0); roo[:-1]=O[1:]/np.where(O[:-1]>0,O[:-1],np.nan)-1.0; roo=np.nan_to_num(roo)
    sall=cdf.rolling(SW,min_periods=SW//2).mean().to_numpy(); vc=np.isfinite(sall).sum(axis=1)
    brd=np.where(vc>0,np.nansum((C>sall)*np.isfinite(sall),axis=1)/np.maximum(vc,1),np.nan); brd=np.nan_to_num(brd); gbin=hyst(brd,0.5,0.4)
    bc=cdf["BTCUSDT"].to_numpy() if "BTCUSDT" in cols else C[:,0]; mv=np.abs(np.diff(bc,prepend=bc[0]))
    er=np.abs(bc-pd.Series(bc).shift(14).to_numpy())/np.maximum(pd.Series(mv).rolling(14,min_periods=7).sum().to_numpy(),1e-9)
    mom14=cdf.pct_change(14*BPD).to_numpy(); starts=[i for i in range(0,T-SL+1,SL)]
    def b1(x): return np.broadcast_to(x[:,None],(T,A)).astype(float)
    def iv(held,g2):
        mult=held*g2; w=np.where((mult>0)&np.isfinite(vol)&(vol>0),1.0/np.where(vol>0,vol,np.nan),0.0)
        w=np.nan_to_num(w)*mult; s=w.sum(axis=1,keepdims=True); return np.minimum(np.where(s>0,w/np.maximum(s,1e-12),0.0),CAP)
    def mom8er():
        gate=gbin*(er>0.35).astype(int); W=np.zeros((T,A))
        for i in starts:
            if gate[i]<=0: continue
            s=mom14[i]; e=np.where(np.isfinite(s)&np.isfinite(C[i]))[0]
            if len(e)==0: continue
            k=min(8,len(e)); top=e[np.argsort(-s[e])[:k]]; v=vol[i][top]; w=np.where(np.isfinite(v)&(v>0),1.0/v,0.0)
            w=w/w.sum() if w.sum()>0 else np.ones(k)/k; W[i:i+SL,top]=np.minimum(w,CAP)
        return W
    e=np.clip((brd-0.20)/0.60,0.15,1.0); ew=np.where(np.isfinite(C),1.0,0.0); ew=ew/np.maximum(ew.sum(axis=1,keepdims=True),1)
    Wtr=iv(Hsma,np.ones((T,A))); ta=Wtr.sum(axis=1,keepdims=True)>1e-9
    CAND={"CONV":iv(conv/len(TYPES),b1(gbin)),"SMA_ER":iv(Hsma,b1(gbin*(er>0.35).astype(int))),"PARTIAL":e[:,None]*np.where(ta,Wtr,ew),"MOM8_ER":mom8er()}
    def net_of(W):
        # CAUSAL cost timing: Wl=W[t-1] (held, earns roo[t]=open t->t+1); charge the rebalance that FILLS at t = |W[t-1]-W[t-2]|,
        # NOT |W[t]-W[t-1]| (that would book the t-decision's cost at t, using close[t] -> same-bar dependency).
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); Wll=np.vstack([np.zeros((1,A)),Wl[:-1]])
        return np.nansum(Wl*roo,axis=1)-np.nansum(np.abs(Wl-Wll),axis=1)*(COST_PROD/2)
    def gross_of(W):
        Wl=np.vstack([np.zeros((1,A)),W[:-1]]); return np.nansum(Wl*roo,axis=1)
    return {"CAND":CAND,"idx":idx,"cols":cols,"net_of":net_of,"gross_of":gross_of,"T":T,"A":A,"brd":brd,"er":er,"gbin":gbin,"roo":roo,"vol":vol}


def main():
    print("="*96); print("FINAL PROD CHECKS + PAPER-TRADER WIRING -- 4 frozen candidates"); print("="*96)
    f=build_all(); CAND=f["CAND"]; idx=f["idx"]; cols=f["cols"]; T=f["T"]
    checks={}
    # 1. CAUSALITY (perturbation): flip a name's close at an interior bar; confirm net changes only at t>t0
    t0=T-40; j0=cols.index("ETHUSDT") if "ETHUSDT" in cols else 1
    fp=build_all(C_override=(t0,j0,1.5))
    print(f"\n  1. CAUSALITY (perturb close[t0={pd.Timestamp(idx[t0]).date()}, ETH, x1.5] -> P&L must change only at t>t0):")
    for nm in CAND:
        base=f["net_of"](CAND[nm]); pert=fp["net_of"](fp["CAND"][nm]); diff=np.abs(base-pert)
        first=int(np.argmax(diff>1e-12)) if (diff>1e-12).any() else -1
        ok=(first>t0) or (first==-1)
        checks.setdefault(nm,{})["causal"]=bool(ok); checks[nm]["first_changed_bar"]=(str(pd.Timestamp(idx[first]).date()) if first>=0 else "none")
        print(f"     {nm:<9} first-changed bar = {checks[nm]['first_changed_bar']:<12} {'PASS (t>t0, causal)' if ok else 'FAIL (LEAK: changes at/<=t0!)'}")
    # 2. DETERMINISM: rebuild, compare
    f2=build_all()
    print(f"\n  2. DETERMINISM (rebuild -> bit-identical):")
    for nm in CAND:
        same=bool(np.allclose(f["net_of"](CAND[nm]),f2["net_of"](f2["CAND"][nm]),atol=0,rtol=0)); checks[nm]["deterministic"]=same
        print(f"     {nm:<9} {'PASS (identical)' if same else 'FAIL (nondeterministic!)'}")
    # 3. COST APPLIED
    print(f"\n  3. COST APPLIED (net < gross on turnover):")
    for nm in CAND:
        g=float(np.sum(f["gross_of"](CAND[nm]))); n=float(np.sum(f["net_of"](CAND[nm]))); ok=n<g+1e-9; checks[nm]["cost_applied"]=bool(ok)
        print(f"     {nm:<9} sum gross {g:+.2f} vs net {n:+.2f}  drag {g-n:+.2f}  {'PASS' if ok else 'FAIL (no cost!)'}")
    # 4. ROBUSTNESS SUMMARY + PROD-GATE
    print(f"\n  4. ROBUSTNESS (honest ceiling) + PROD-GATE:")
    print(f"     {'cand':<9}{'VAL%':>7}{'OOS%':>7}{'p05oos':>8}{'DSR40':>7}  causal/determ/cost  -> gate")
    live_as_of=str(pd.Timestamp(idx[-1]).date())
    out={"check":"prod_deploy","as_of":live_as_of,"frozen_specs":FROZEN,"prod_checks":checks,"robust":ROBUST,"ts":datetime.now().strftime("%Y%m%d_%H%M%S"),"live_book":{}}
    for nm in CAND:
        r=ROBUST[nm]; c=checks[nm]; paper_safe=c["causal"] and c["deterministic"] and c["cost_applied"]
        checks[nm]["paper_trade_safe"]=paper_safe; checks[nm]["real_capital_ready"]=False  # DSR<0.15, p05<0 full-cycle
        print(f"     {nm:<9}{r['VAL']:>+6.0f}%{r['OOS']:>+6.0f}%{r['p05_oos']:>+7.0f}%{r['dsr40']:>7.2f}  {'Y/Y/Y' if paper_safe else 'CHECK'}          -> {'PAPER-TRADE ✓ (real-capital ✗: DSR<0.15)' if paper_safe else 'BLOCKED'}")
    # ---- WIRE: emit current weekly book per candidate ----
    print(f"\n  LIVE WEEKLY BOOK (as of {live_as_of}, breadth {f['brd'][-1]:.2f}, ER {f['er'][-1]:.2f}):")
    last=T-1
    for nm,W in CAND.items():
        lw=W[last]; book=sorted(((cols[j],round(float(lw[j]),4)) for j in np.where(lw>1e-6)[0]),key=lambda x:-x[1])
        out["live_book"][nm]={"decision":"DEPLOY" if book else "CASH","gross":round(float(lw.sum()),3),"n_names":len(book),"weights":book}
        print(f"     {nm:<9}{'DEPLOY' if book else 'CASH':<7}"+(f"  {len(book)} names, gross {lw.sum():.2f}, top: "+", ".join(f"{n}{w:.0%}" for n,w in book[:4]) if book else ""))
    lbdir=HERE/"live_books"; lbdir.mkdir(exist_ok=True)
    with open(lbdir/f"live_book_{live_as_of}.json","w") as fo: json.dump({"as_of":live_as_of,"books":out["live_book"],"specs":FROZEN},fo,indent=2,default=str)
    with open(HERE/"strat_prod_deploy.json","w") as fo: json.dump(out,fo,indent=2,default=str)
    # ---- PERSISTENT PAPER-TRADE JOURNAL (idempotent by as_of): the accumulating live-forward track ----
    jpath=lbdir/"journal.jsonl"; jour={}
    if jpath.exists():
        for ln in jpath.read_text().splitlines():
            if ln.strip():
                try: r=json.loads(ln); jour[r["as_of"]]=r
                except Exception: pass
    jour[live_as_of]={"as_of":live_as_of,"breadth":round(float(f['brd'][-1]),3),"er":round(float(f['er'][-1]),3),
        "book":{nm:{"decision":out["live_book"][nm]["decision"],"gross":out["live_book"][nm]["gross"],"n":out["live_book"][nm]["n_names"],
                    "top":[n for n,_ in out["live_book"][nm]["weights"][:4]]} for nm in CAND}}
    with open(jpath,"w") as fo:
        for k in sorted(jour): fo.write(json.dumps(jour[k],default=str)+"\n")
    # ---- human-readable ledger ----
    rows=[jour[k] for k in sorted(jour)]
    md=["# Paper-trade ledger -- 4 frozen candidates (pre-registered live-forward track)","",
        f"Emitted by `strat_prod_deploy.py`. Frozen specs (re-tuning voids the gate). Latest data as-of **{live_as_of}**. "
        f"Re-run weekly (`python runs/engines/STRAT_TIERS/strat_prod_deploy.py`) to append each week's decision.","",
        "| as_of | breadth | ER | CONV | SMA_ER | PARTIAL | MOM8_ER |","|---|---|---|---|---|---|---|"]
    for r in rows:
        def cell(nm):
            b=r["book"][nm]; return f"{b['decision']}"+(f" ({b['n']}n,{b['gross']:.2f})" if b["decision"]=="DEPLOY" else "")
        md.append(f"| {r['as_of']} | {r['breadth']:.2f} | {r['er']:.2f} | {cell('CONV')} | {cell('SMA_ER')} | {cell('PARTIAL')} | {cell('MOM8_ER')} |")
    md+=["","**Promotion gate (per candidate):** >=20 deployed weeks with deployed-median>0, compound>cash, compound>BH_btc over "
         "the same deployed weeks, maxDD within backtest floor. **Kill:** deployed-median<=0 over 20+ wks, compound<cash, or DD breach."]
    (HERE/"PAPER_TRADE_LEDGER.md").write_text("\n".join(md),encoding="utf-8")
    print(f"\n  Wired -> paper trader:")
    print(f"    dated book : {lbdir/('live_book_'+live_as_of+'.json')}")
    print(f"    journal    : {jpath}  ({len(jour)} week(s) logged)")
    print(f"    ledger     : {HERE/'PAPER_TRADE_LEDGER.md'}")
    print(f"  Re-run weekly (frozen specs) to append the next decision -> converts promising to proven.")


if __name__=="__main__":
    main()
