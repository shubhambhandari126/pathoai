/* PathoAI — pathologist/intern app (single-file React).
   Main flow is ONE guided path: upload a case -> see the whole-slide overview and
   verdict -> zoom into the flagged patches -> confirm the read (which also teaches
   the model). The batch worklists live under a secondary "Review queue" tab.
   PathoAI never changes a sign-out and never diagnoses. */
const {useState, useEffect, useCallback} = React;

const POST = (u, b) => fetch(u, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(b)}).then(r=>r.json());
const api = {
  summary: () => fetch("/api/summary").then(r=>r.json()),
  worklist: () => fetch("/api/worklist").then(r=>r.json()),
  settings: (b) => POST("/api/settings", b),
  decision: (b) => POST("/api/decision", b),
  slideSummary: () => fetch("/api/slides/summary").then(r=>r.json()),
  slideWorklist: () => fetch("/api/slides/worklist").then(r=>r.json()),
  slideSettings: (b) => POST("/api/slides/settings", b),
  slideDecision: (b) => POST("/api/slides/decision", b),
  analyzeImage: (image) => POST("/api/analyze-image", {image}),
  feedback: (image, label, autoLearn) => POST("/api/feedback", {image, label, auto_learn: !!autoLearn}),
  feedbackStats: () => fetch("/api/feedback/stats").then(r=>r.json()),
  loadModel: (model) => POST("/api/load-model", {model}),
  modelInfo: () => fetch("/api/model-info").then(r=>r.json()),
  retrain: (epochs) => POST("/api/retrain", {epochs}),
};

const pct = (x) => (100*x).toFixed(1) + "%";
const Read = ({v}) => <span className={"pill " + v}>{v}</span>;
const Card = ({k, v, sub}) => <div className="card"><div className="k">{k}</div>
  <div className="v">{v} {sub && <small>{sub}</small>}</div></div>;
const readLabel = (r)=> r==="tumor"?"tumour":r;
const readPill = (r)=> <span className={"pill "+(r==="tumor"?"tumor":r==="normal"?"normal":"uncertain")}>{readLabel(r)}</span>;
const IMG = {width:150,height:150,objectFit:"cover",border:"1px solid var(--line)",borderRadius:10};

/* ---- text report classifier (client-side, 5 buckets, negation-aware) ---- */
const CATS = {
  benign:{label:"benign / negative",rank:0,bg:"#0f241d",fg:"#a7f3d0"},
  atypical:{label:"atypical / suspicious",rank:1,bg:"#2a1c0a",fg:"#fcd34d"},
  dysplasia:{label:"dysplasia / pre-malignant",rank:2,bg:"#2a1c0a",fg:"#fcd34d"},
  malignant:{label:"malignant / carcinoma",rank:3,bg:"#2a1414",fg:"#fca5a5"},
  nondiag:{label:"non-diagnostic / insufficient",rank:null,bg:"#1c242e",fg:"#8b98a5"},
  uncertain:{label:"uncertain",rank:null,bg:"#1c242e",fg:"#8b98a5"},
};
const catPill = (c)=>{ const x=CATS[c]||CATS.uncertain;
  return <span style={{background:x.bg,color:x.fg,padding:"3px 10px",borderRadius:999,fontSize:13,fontWeight:600}}>{x.label}</span>; };
function scoreCats(t){
  t=" "+(t||"").toLowerCase().replace(/\s+/g," ")+" ";
  const s={malignant:0,dysplasia:0,atypical:0,benign:0,nondiag:0}; const terms=[];
  const add=(k,w,lab,side)=>{s[k]+=w; terms.push({label:lab,w,side});};
  [[/non[- ]?diagnostic/g,2,"non-diagnostic"],[/insufficient|inadequate|scant|paucicellular|acellular/g,1.5,"insufficient sample"],[/cannot be (assessed|evaluated|interpreted)/g,1.5,"cannot be assessed"],[/no (epithelial|diagnostic) (cells|material)/g,1.5,"no diagnostic material"]].forEach(([re,w,lab])=>{const m=t.match(re); if(m)add("nondiag",w*m.length,lab,"nd");});
  const negMal=/(no|negative for|without|absence of|free of)\s+(evidence of\s+)?((?:\w+\s+){0,2})(malignan\w*|tumou?rs?|carcinoma|dysplasi\w*|atypi\w*|neoplas\w*)/g;
  const neg=t.match(negMal); if(neg){neg.forEach(x=>add("benign",3,'"'+x.trim()+'"',"ben")); t=t.replace(negMal," ");}
  [[/benign/g,2.5,"benign"],[/within normal limits/g,2.5,"within normal limits"],[/unremarkable/g,1.5,"unremarkable"],[/reactive/g,1,"reactive changes"],[/preserved (crypt )?architecture/g,1.5,"preserved architecture"]].forEach(([re,w,lab])=>{const m=t.match(re); if(m)add("benign",w*m.length,lab,"ben");});
  [[/adenocarcinoma/g,4,"adenocarcinoma"],[/carcinoma/g,3,"carcinoma"],[/sarcoma/g,3,"sarcoma"],[/malignan\w*/g,3,"malignant"],[/invasiv\w*|invasion/g,2,"invasion"],[/metasta\w*/g,3,"metastasis"]].forEach(([re,w,lab])=>{const m=t.match(re); if(m)add("malignant",w*m.length,lab,"mal");});
  [[/high[- ]grade dysplasia/g,3,"high grade dysplasia"],[/low[- ]grade dysplasia/g,2,"low grade dysplasia"],[/dysplas\w*/g,1.5,"dysplasia"],[/intraepithelial neoplasi\w*/g,2,"intraepithelial neoplasia"]].forEach(([re,w,lab])=>{const m=t.match(re); if(m)add("dysplasia",w*m.length,lab,"warn");});
  [[/atypi\w*/g,1.5,"atypia"],[/suspicious/g,2,"suspicious"],[/cannot exclude|cannot rule out/g,1.5,"cannot exclude malignancy"],[/indeterminate/g,1.5,"indeterminate"]].forEach(([re,w,lab])=>{const m=t.match(re); if(m)add("atypical",w*m.length,lab,"warn");});
  return {s,terms};
}
function classifyReport(t){
  const base=scoreCats(t); const s={...base.s};
  const idx=(t||"").toLowerCase().search(/impression|diagnosis|conclusion/);
  if(idx>=0){const si=scoreCats(t.slice(idx)).s; for(const k in s) s[k]+=2*si[k];}
  const axis=["malignant","dysplasia","atypical","benign"];
  const total=Object.values(s).reduce((a,b)=>a+b,0);
  if(total===0) return {cat:"uncertain",conf:0.45,terms:base.terms};
  const axisSum=axis.reduce((a,k)=>a+s[k],0);
  if(s.nondiag>0 && s.nondiag>=axisSum) return {cat:"nondiag",conf:Math.min(0.95,s.nondiag/total+0.3),terms:base.terms};
  const sorted=axis.map(k=>[k,s[k]]).sort((a,b)=>b[1]-a[1]);
  const topK=sorted[0][0], topV=sorted[0][1], secV=sorted[1][1];
  if(topV===0) return {cat:"uncertain",conf:0.5,terms:base.terms};
  return {cat:topK,conf:Math.min(0.96,Math.max(0.55,0.55+0.41*(topV-secV)/(axisSum||1))),terms:base.terms};
}

function Actions({onConfirm, onAmend, note, setNote}) {
  return <div style={{display:"flex",gap:6,alignItems:"center",flexWrap:"wrap"}}>
    <button className="btn confirm" onClick={onConfirm}>Confirm sign-out</button>
    <button className="btn amend" onClick={onAmend}>Amend (model was right)</button>
    <input placeholder="note…" value={note||""} onChange={e=>setNote(e.target.value)}
      style={{background:"var(--panel2)",border:"1px solid var(--line)",color:"var(--text)",borderRadius:6,padding:"5px 7px",fontSize:12,width:110}}/>
  </div>;
}
const Tag = ({d}) => <span className={"tag " + d}>{d === "amend" ? "Amended" : "Sign-out confirmed"}</span>;

/* =================== GUIDED FLOW (the front door) =================== */
function Step({n, title, children}) {
  return <div className="card" style={{display:"flex",gap:12,alignItems:"flex-start"}}>
    <div style={{flex:"none",width:26,height:26,borderRadius:"50%",background:"var(--accent)",color:"#0b1418",
      fontWeight:700,fontSize:13,display:"flex",alignItems:"center",justifyContent:"center",marginTop:1}}>{n}</div>
    <div style={{flex:1}}>{title && <div style={{fontSize:15,fontWeight:600,marginBottom:8}}>{title}</div>}{children}</div>
  </div>;
}

function GuidedFlow() {
  const [img, setImg] = useState(null);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [stats, setStats] = useState(null);
  const [decision, setDecision] = useState(null);
  const [zoom, setZoom] = useState(null);
  const [modelMsg, setModelMsg] = useState("Checking model…");
  const [retrainMsg, setRetrainMsg] = useState("");

  useEffect(() => { api.feedbackStats().then(setStats);
    api.modelInfo().then(mi => {
      const a = mi.metrics && mi.metrics.test_auroc;
      if (mi.source === "trained") setModelMsg("Using the bundled trained model" + (a ? " · AUROC " + a.toFixed(2) : "") + ".");
      else if (mi.source === "uploaded") setModelMsg("Using your loaded model" + (a ? " · AUROC " + a.toFixed(2) : "") + ".");
      else setModelMsg("Using the built-in demo model (train and load one for real reads).");
    }).catch(()=>setModelMsg("")); }, []);
  const loadModel = (e) => { const f=e.target.files[0]; if(!f)return; setModelMsg("Loading "+f.name+"…");
    const r=new FileReader(); r.onload=async()=>{ try{ const res=await api.loadModel(r.result);
      const a=res.metrics&&res.metrics.test_auroc; setModelMsg("Loaded "+f.name+(a?" · test AUROC "+a.toFixed(2):"")+". Analyze an image to use it.");
    }catch(err){ setModelMsg("Could not load that file."); } }; r.readAsDataURL(f); };
  const pick = (e) => { const f=e.target.files[0]; if(!f)return; const r=new FileReader();
    r.onload=()=>{ setImg(r.result); setRes(null); setDecision(null); }; r.readAsDataURL(f); };
  const run = async () => { if(!img)return; setBusy(true); setRes(null); setDecision(null);
    try{ setRes(await api.analyzeImage(img)); } finally{ setBusy(false); } };
  const confirmRead = async (label) => {
    const matched = res.combined.read===label;
    setDecision({label, matched, saving:true});
    const r = await api.feedback(img, label, true);   // auto-learn on
    if (r.stats) setStats(r.stats);
    setDecision({label, matched, learned: r.retrained, auroc: r.val_auroc});
    if (r.retrained) setModelMsg("Model updated from submissions"+(r.val_auroc?" · val AUROC "+r.val_auroc:"")+" (saved).");
  };
  const reset = () => { setImg(null); setRes(null); setDecision(null); };
  const retrain = async () => { setRetrainMsg("Fine-tuning on your submitted labels… this can take a minute.");
    try { const r = await api.retrain(3);
      setModelMsg("Retrained on your labels · val AUROC " + r.val_auroc + ". Now in use.");
      setRetrainMsg("Done. Trained on " + r.n_train + " images; updated model saved and live.");
      api.feedbackStats().then(setStats);
    } catch(e) { setRetrainMsg("Need at least 3 'normal' and 3 'tumour' labels before fine-tuning."); } };

  const verdictColor = (r)=> r==="tumor"?"var(--tumor,#f87171)":r==="normal"?"var(--ok,#34d399)":"var(--warn,#f59e0b)";

  return <>
    <div className="card" style={{display:"flex",alignItems:"center",gap:10,flexWrap:"wrap",padding:"10px 14px"}}>
      <span style={{fontSize:13,color:"var(--muted)"}}>{modelMsg}</span>
      <label className="btn" style={{cursor:"pointer",marginLeft:"auto"}}>Load model.pt
        <input type="file" accept=".pt" style={{display:"none"}} onChange={loadModel}/></label>
    </div>
    <Step n="1" title="Upload a case">
      <div style={{fontSize:13,color:"var(--muted)",marginBottom:8}}>Choose an H&E image (a small patch or a whole slide). Nothing leaves this machine.</div>
      <input type="file" accept="image/*" onChange={pick}/>
      {img && <button className="btn primary" style={{marginLeft:10}} disabled={busy} onClick={run}>{busy?"Analyzing…":"Analyze case"}</button>}
      {img && !res && <div style={{marginTop:10}}><img src={img} style={IMG}/></div>}
    </Step>

    {res && <>
      <Step n="2" title="PathoAI's read">
        <div className="card" style={{borderLeft:"4px solid "+verdictColor(res.combined.read),marginTop:0}}>
          <div style={{fontSize:18,fontWeight:600}}>{readPill(res.combined.read)} <span style={{fontSize:13,color:"var(--muted)",fontWeight:400}}>combined verdict · {(100*res.combined.confidence).toFixed(0)}% · {res.combined.agree?"model + stain agree":"needs review"}</span></div>
          <div style={{fontSize:13,color:"var(--muted)",marginTop:6}}>{res.combined.note}</div>
        </div>
        <div className="cards" style={{gridTemplateColumns:"repeat(2,1fr)",marginTop:12}}>
          <div className="card"><div className="k">Trained model</div>
            <div className="v" style={{fontSize:17}}>{readPill(res.model.read)}</div>
            <div style={{fontSize:12,color:"var(--muted)",marginTop:6}}>P(tumour) {(100*res.model.prob_tumor).toFixed(0)}%{res.model.uncertainty!=null?` · uncertainty ${(100*res.model.uncertainty).toFixed(0)}%`:""}</div></div>
          <div className="card"><div className="k">Stain check</div>
            <div className="v" style={{fontSize:17}}>{(100*res.stain.suspicion).toFixed(0)}%</div>
            <div style={{fontSize:12,color:"var(--muted)",marginTop:6}}>dense nuclei {(100*res.stain.frac).toFixed(0)}% of tissue</div></div>
        </div>
        <div style={{display:"flex",gap:14,flexWrap:"wrap",marginTop:12}}>
          <figure style={{margin:0,textAlign:"center"}}><img src={img} style={IMG}/><figcaption style={{fontSize:11,color:"var(--muted)"}}>uploaded</figcaption></figure>
          <figure style={{margin:0,textAlign:"center"}}><img src={res.overlay} style={IMG}/><figcaption style={{fontSize:11,color:"var(--muted)"}}>{res.mode==="slide"?"tumour heatmap":"model attention"}</figcaption></figure>
          <figure style={{margin:0,textAlign:"center"}}><img src={res.stain_overlay} style={IMG}/><figcaption style={{fontSize:11,color:"var(--muted)"}}>stain highlight</figcaption></figure>
        </div>
      </Step>

      {res.mode==="slide" && res.tiles && res.tiles.length>0 &&
        <Step n="3" title="Zoom into the most-suspicious regions">
          <div style={{fontSize:13,color:"var(--muted)",marginBottom:10}}>The specific patches the model reacted to most, with what it looked at (Grad-CAM). This is the "show me where" step.</div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(150px,1fr))",gap:12}}>
            {res.tiles.map((t,i)=><div key={i} className="card" style={{padding:10}}>
              <div style={{display:"flex",gap:6}}>
                <img src={t.tile} style={{width:"50%",borderRadius:6,border:"1px solid var(--line)"}}/>
                <img src={t.cam} style={{width:"50%",borderRadius:6,border:"1px solid var(--line)"}}/>
              </div>
              <div style={{fontSize:11,color:"var(--muted)",marginTop:6}}>tile ({t.row},{t.col}) · P(tumour) {(100*t.prob).toFixed(0)}%</div>
            </div>)}
          </div>
        </Step>}

      <Step n={res.mode==="slide"?"4":"3"} title="Your call">
        {!decision ? <>
          <div style={{fontSize:13,color:"var(--muted)",marginBottom:8}}>What is your read? This is recorded, and your answer becomes a training example so the model improves.</div>
          <button className="btn confirm" onClick={()=>confirmRead("normal")}>This is normal</button>
          <button className="btn amend" style={{marginLeft:8}} onClick={()=>confirmRead("tumor")}>This is tumour (cancer)</button>
        </> : <>
          <div style={{fontSize:14}}>You marked {readPill(decision.label)}. {decision.matched
            ? <span style={{color:"var(--ok,#34d399)"}}>PathoAI agreed, no discrepancy.</span>
            : <span style={{color:"var(--warn,#f59e0b)"}}>This differs from PathoAI's read, exactly the kind of case worth a second look. Your correction now trains it.</span>}</div>
          {decision.saving && <div style={{fontSize:12,color:"var(--muted)",marginTop:8}}>Saving and updating the model…</div>}
          {decision.learned && <div style={{fontSize:12,color:"var(--ok,#34d399)",marginTop:8}}>Saved and folded into the model{decision.auroc?` (val AUROC ${decision.auroc})`:""}. It will use this from the next case on.</div>}
          {!decision.saving && !decision.learned && stats && (stats.normal<3||stats.tumor<3) &&
            <div style={{fontSize:12,color:"var(--muted)",marginTop:8}}>Saved. Auto-learning kicks in once there are 3+ of each label (have {stats.normal} normal, {stats.tumor} tumour).</div>}
          <div style={{marginTop:12}}>
            <button className="btn primary" onClick={reset}>Analyze another case</button>
          </div>
        </>}
      </Step>
    </>}

    {zoom && <div className="modal" onClick={()=>setZoom(null)}><div className="box" onClick={e=>e.stopPropagation()}>
      <img src={zoom} alt="zoom"/></div></div>}
  </>;
}

/* =================== Batch worklists (secondary "Review queue") =================== */
function PatchesView() {
  const [summary, setSummary] = useState(null);
  const [items, setItems] = useState([]);
  const [tau, setTau] = useState(0.90);
  const [zoom, setZoom] = useState(null);
  const [notes, setNotes] = useState({});

  const refresh = useCallback(async () => {
    const [s, w] = await Promise.all([api.summary(), api.worklist()]);
    setSummary(s); setItems(w.items || []);
    if (s && typeof s.tau === "number") setTau(s.tau);
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  const applyTau = async (v) => { setTau(v); const s = await api.settings({tau:v});
    if (s.summary) setSummary(s.summary); const w = await api.worklist(); setItems(w.items||[]); };
  const decide = async (id, a) => { await api.decision({case_id:id, action:a, note:notes[id]||""}); refresh(); };

  if (!summary) return <div className="empty">Loading patch QA…</div>;
  const pending = items.filter(i => !i.decision);

  return <>
    <div className="cards">
      <Card k="Cases reviewed" v={summary.n_cases} />
      <Card k="Disagreements" v={summary.n_disagreements} sub={"· " + pct(summary.disagreement_rate)} />
      <Card k="Flagged for re-check" v={summary.n_flagged} sub={"· " + pct(summary.flag_rate)} />
      <Card k="Resolved" v={summary.resolved + " / " + summary.n_flagged} />
    </div>
    <div className="controls">
      <label>Flag confidence threshold (τ)</label>
      <input type="range" min="0.5" max="0.999" step="0.005" value={tau}
        onChange={e=>setTau(parseFloat(e.target.value))}
        onMouseUp={e=>applyTau(parseFloat(e.target.value))}
        onTouchEnd={e=>applyTau(parseFloat(e.target.value))} />
      <span className="mono">τ = {Number(tau).toFixed(3)}</span>
      <span style={{color:"var(--muted)",fontSize:12}}>lower τ → catch more · higher τ → shorter queue</span>
    </div>
    {pending.length === 0
      ? <div className="empty">Nothing left to re-check at this threshold. ✅</div>
      : <table><thead><tr>
          <th>#</th><th>Patch</th><th>Case</th><th>Sign-out</th><th>Model reads</th><th>Confidence</th><th>Re-review outcome</th>
        </tr></thead><tbody>
          {items.map(it => <tr key={it.case_id} className={it.decision?"done":""}>
            <td>{it.priority}</td>
            <td><img className="thumb" src={"/api/patch/"+it.case_id+".png"} onClick={()=>setZoom(it)} alt="patch"/></td>
            <td className="mono">{it.case_id}</td>
            <td><Read v={it.pathologist_read}/></td>
            <td><Read v={it.model_read}/></td>
            <td><div className="confbar"><span style={{width:(100*it.model_confidence)+"%"}}/></div>
              <span className="mono" style={{fontSize:11}}>{it.model_confidence.toFixed(3)}</span></td>
            <td>{it.decision ? <Tag d={it.decision}/> :
              <Actions onConfirm={()=>decide(it.case_id,"confirm_signout")} onAmend={()=>decide(it.case_id,"amend")}
                note={notes[it.case_id]} setNote={v=>setNotes({...notes,[it.case_id]:v})}/>}</td>
          </tr>)}
        </tbody></table>}
    {zoom && <div className="modal" onClick={()=>setZoom(null)}><div className="box" onClick={e=>e.stopPropagation()}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:10}}>
        <span className="mono">{zoom.case_id}</span>
        <span style={{cursor:"pointer",color:"var(--muted)"}} onClick={()=>setZoom(null)}>✕</span></div>
      <div style={{display:"flex",gap:10}}>
        <div style={{flex:1,textAlign:"center"}}><img src={"/api/patch/"+zoom.case_id+".png"} alt="patch"/>
          <div style={{fontSize:11,color:"var(--muted)",marginTop:4}}>patch</div></div>
        <div style={{flex:1,textAlign:"center"}}><img src={"/api/patch/"+zoom.case_id+"/cam.png"} alt="grad-cam"/>
          <div style={{fontSize:11,color:"var(--muted)",marginTop:4}}>model attention (Grad-CAM)</div></div>
      </div>
      <p style={{fontSize:13,color:"var(--muted)",marginTop:12}}>{zoom.reason}</p></div></div>}
  </>;
}

function SlidesView() {
  const [summary, setSummary] = useState(null);
  const [items, setItems] = useState([]);
  const [tau, setTau] = useState(0.70);
  const [notes, setNotes] = useState({});
  const [zoom, setZoom] = useState(null);

  const refresh = useCallback(async () => {
    const [s, w] = await Promise.all([api.slideSummary(), api.slideWorklist()]);
    setSummary(s); setItems(w.items || []);
    if (s && typeof s.slide_tau === "number") setTau(s.slide_tau);
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  const applyTau = async (v) => { setTau(v); await api.slideSettings({tau:v}); refresh(); };
  const decide = async (id, a) => { await api.slideDecision({slide_id:id, action:a, note:notes[id]||""}); refresh(); };

  if (!summary) return <div className="empty">Loading slide QA…</div>;

  return <>
    <div className="cards" style={{gridTemplateColumns:"repeat(3,1fr)"}}>
      <Card k="Slides reviewed" v={summary.n_slides} />
      <Card k="Flagged for re-review" v={summary.n_flagged} />
      <Card k="Resolved" v={summary.resolved + " / " + summary.n_flagged} />
    </div>
    <div className="controls">
      <label>Slide flag threshold (τ)</label>
      <input type="range" min="0.5" max="0.999" step="0.005" value={tau}
        onChange={e=>setTau(parseFloat(e.target.value))}
        onMouseUp={e=>applyTau(parseFloat(e.target.value))}
        onTouchEnd={e=>applyTau(parseFloat(e.target.value))} />
      <span className="mono">τ = {Number(tau).toFixed(3)}</span>
      <span style={{color:"var(--muted)",fontSize:12}}>heatmap: green = normal, red = suspicious tissue</span>
    </div>
    {items.length === 0
      ? <div className="empty">No slides flagged at this threshold. ✅</div>
      : <div className="slidegrid">
        {items.map(it => <div key={it.slide_id} className={"slidecard" + (it.decision?" done":"")}>
          <img className="heat" src={"/api/slide/"+it.slide_id+"/heatmap.png"} onClick={()=>setZoom(it)} alt="heatmap"/>
          <div className="scbody">
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
              <span className="mono">#{it.priority} · {it.slide_id}</span>
              <span className="mono" style={{fontSize:11,color:"var(--muted)"}}>score {it.slide_score.toFixed(2)}</span>
            </div>
            <div style={{margin:"8px 0",fontSize:13}}>sign-out <Read v={it.pathologist_read}/> &nbsp;vs&nbsp; model <Read v={it.model_read}/></div>
            <div style={{fontSize:12,color:"var(--muted)",marginBottom:10}}>{it.n_tumor_tiles} suspicious / {it.n_tissue_tiles} tissue tiles · conf {it.confidence.toFixed(2)}</div>
            {it.decision ? <Tag d={it.decision}/> :
              <Actions onConfirm={()=>decide(it.slide_id,"confirm_signout")} onAmend={()=>decide(it.slide_id,"amend")}
                note={notes[it.slide_id]} setNote={v=>setNotes({...notes,[it.slide_id]:v})}/>}
          </div>
        </div>)}
      </div>}
    {zoom && <div className="modal" onClick={()=>setZoom(null)}><div className="box" onClick={e=>e.stopPropagation()}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:10}}>
        <span className="mono">{zoom.slide_id}</span>
        <span style={{cursor:"pointer",color:"var(--muted)"}} onClick={()=>setZoom(null)}>✕</span></div>
      <img src={"/api/slide/"+zoom.slide_id+"/heatmap.png"} alt="heatmap"/>
      <p style={{fontSize:13,color:"var(--muted)",marginTop:12}}>{zoom.reason}</p></div></div>}
  </>;
}

function ReviewQueue() {
  const [q, setQ] = useState("patches");
  return <>
    <div className="tabs" style={{marginBottom:12}}>
      <button className={"tab"+(q==="patches"?" active":"")} onClick={()=>setQ("patches")}>Patches</button>
      <button className={"tab"+(q==="slides"?" active":"")} onClick={()=>setQ("slides")}>Whole slides</button>
    </div>
    <div style={{fontSize:12,color:"var(--muted)",marginBottom:10}}>Demo review queues: a batch of already signed-out cases; PathoAI lists only the ones it disagrees with.</div>
    {q==="patches" ? <PatchesView/> : <SlidesView/>}
  </>;
}

/* =================== Report check (text, client-side) =================== */
const FIELD = {width:"100%",background:"var(--panel2)",border:"1px solid var(--line)",color:"var(--text)",borderRadius:8,padding:"9px 11px",fontSize:14,fontFamily:"inherit"};
const EXAMPLE_RPT = "SPECIMEN: Sigmoid colon, biopsy.\nMICROSCOPIC: Irregular cribriform glands with hyperchromatic, pleomorphic nuclei and desmoplastic stroma; foci suggest submucosal invasion.\nIMPRESSION: Findings consistent with invasive adenocarcinoma.";
function ReportCheck() {
  const [report, setReport] = useState("");
  const [concl, setConcl] = useState(null);
  const [free, setFree] = useState("");
  const [res, setRes] = useState(null);

  const run = () => { if(!report.trim()){ setRes({error:true}); return; } setRes(classifyReport(report)); };
  const patho = free.trim() ? classifyReport(free).cat : concl;
  const segBtn = (id) => <button key={id} onClick={()=>{setConcl(id); setFree("");}}
    style={{textAlign:"left",border:"1.5px solid "+(concl===id?"var(--accent)":"var(--line)"),
      background:concl===id?"var(--panel2)":"var(--panel)",color:"var(--text)",borderRadius:10,padding:"9px 11px",cursor:"pointer",fontSize:13}}>
    {CATS[id].label}</button>;

  const verdict = () => {
    if(!res || res.error) return null;
    const a=res.cat;
    if(!patho) return <div className="card" style={{borderLeft:"4px solid #6b7280"}}><b>Read only.</b> Pick or type the pathologist's conclusion to compare.</div>;
    if(a==="uncertain") return <div className="card" style={{borderLeft:"4px solid var(--warn,#f59e0b)"}}><b>PathoAI is not sure.</b> Add more report text or confirm with the pathologist.</div>;
    if(a==="nondiag"||patho==="nondiag"){
      const both=a==="nondiag"&&patho==="nondiag";
      return <div className="card" style={{borderLeft:"4px solid "+(both?"var(--ok,#34d399)":"var(--warn,#f59e0b)")}}>
        {both?<><b>Both say non-diagnostic.</b> Request a better sample.</>:<><b>Adequacy mismatch.</b> Confirm sample adequacy with the pathologist.</>}</div>;
    }
    const gap=Math.abs(CATS[a].rank-CATS[patho].rank);
    const col=gap===0?"var(--ok,#34d399)":gap===1?"var(--warn,#f59e0b)":"var(--tumor,#f87171)";
    return <div className="card" style={{borderLeft:"4px solid "+col}}>
      <div style={{marginBottom:6}}>Pathologist {catPill(patho)} &nbsp;vs&nbsp; PathoAI {catPill(a)}</div>
      {gap===0 ? <><b>Matches.</b> No discrepancy, safe to proceed.</>
        : gap===1 ? <><b>Borderline difference.</b> Worth a quick confirm with the pathologist.</>
        : <><b>Clear discrepancy, second look needed.</b> Take it back to the pathologist before sign-out. Do not change the read yourself.</>}</div>;
  };

  return <>
    <Step n="1" title="Paste the pathologist's report">
      <textarea style={{...FIELD,minHeight:130,resize:"vertical"}} value={report}
        onChange={e=>setReport(e.target.value)} placeholder="Paste the report or impression here…"/>
      <button className="btn" style={{marginTop:8}} onClick={()=>setReport(EXAMPLE_RPT)}>Load example</button>
    </Step>
    <Step n="2" title="What did the pathologist conclude?">
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",gap:8}}>
        {["benign","atypical","dysplasia","malignant","nondiag"].map(segBtn)}
      </div>
      <div style={{fontSize:12,color:"var(--muted)",margin:"10px 0 4px"}}>or type it in their words</div>
      <input style={FIELD} value={free} onChange={e=>{setFree(e.target.value); setConcl(null);}}
        placeholder='e.g. "atypical glandular cells, cannot exclude malignancy"'/>
      {free.trim() && <div style={{fontSize:13,color:"var(--muted)",marginTop:6}}>PathoAI reads your text as {catPill(classifyReport(free).cat)}</div>}
    </Step>
    <Step n="3" title="Check">
      <button className="btn primary" onClick={run}>Run report check</button>
      {res && res.error && <span style={{marginLeft:10,color:"var(--warn,#f59e0b)",fontSize:13}}>Paste a report first.</span>}
    </Step>
    {res && !res.error && <>
      <div className="card">
        <div style={{fontSize:15,fontWeight:600,marginBottom:6}}>PathoAI's independent read: {catPill(res.cat)} <span style={{fontSize:13,color:"var(--muted)",fontWeight:400}}>· {(100*res.conf).toFixed(0)}%</span></div>
        <div style={{fontSize:12,color:"var(--muted)",marginBottom:6}}>Reading the words in the report (not the tissue). Terms it weighted:</div>
        <div>{res.terms.slice().sort((a,b)=>b.w-a.w).map((t,i)=>{const c=t.side==="mal"?CATS.malignant:t.side==="ben"?CATS.benign:t.side==="nd"?CATS.nondiag:CATS.atypical;
          return <span key={i} style={{background:c.bg,color:c.fg,fontSize:12,padding:"3px 9px",borderRadius:999,margin:"4px 6px 0 0",display:"inline-block"}}>{t.label} +{(+t.w).toFixed(1)}</span>;})}
          {res.terms.length===0 && <span style={{fontSize:12,color:"var(--muted)"}}>No clear diagnostic terms found.</span>}</div>
      </div>
      {verdict()}
    </>}
  </>;
}

/* =================== Model details (live database size) =================== */
function ModelInfoView() {
  const [mi, setMi] = useState(null);
  const load = useCallback(() => api.modelInfo().then(setMi), []);
  useEffect(() => { load(); const t=setInterval(load, 5000); return ()=>clearInterval(t); }, [load]);
  if (!mi) return <div className="empty">Loading model info…</div>;
  const auroc = mi.metrics && (mi.metrics.test_auroc || mi.metrics.pcam_auroc || mi.metrics.val_auroc);
  const n = (x)=> (x||0).toLocaleString();
  const srcLabel = {trained:"trained model", uploaded:"your loaded model", retrained:"model (learning from submissions)", demo:"demo model"}[mi.source] || mi.source;
  return <>
    <div className="cards">
      <div className="card"><div className="k">Images in the database</div>
        <div className="v" style={{fontSize:26}}>{n(mi.total_images)}</div>
        <div style={{fontSize:12,color:"var(--muted)",marginTop:6}}>grows by 1 with every scan a pathologist submits</div></div>
      <div className="card"><div className="k">Model score (AUROC)</div>
        <div className="v" style={{fontSize:26}}>{auroc? auroc.toFixed(3):"—"}</div>
        <div style={{fontSize:12,color:"var(--muted)",marginTop:6}}>{srcLabel}</div></div>
    </div>
    <div className="card">
      <div className="stitle" style={{fontSize:15,fontWeight:600,marginBottom:8}}>Where the data comes from</div>
      <table style={{width:"100%",fontSize:13}}>
        <tbody>
          <tr><td style={{padding:"6px 0",color:"var(--muted)"}}>Base training data (PathMNIST + PatchCamelyon)</td><td style={{textAlign:"right"}}>{n(mi.base_images)}</td></tr>
          <tr><td style={{padding:"6px 0",color:"var(--muted)"}}>Pathologist contributions (submitted scans)</td><td style={{textAlign:"right"}}>+ {n(mi.contributions)}</td></tr>
          <tr style={{borderTop:"1px solid var(--line)"}}><td style={{padding:"8px 0",fontWeight:600}}>Total</td><td style={{textAlign:"right",fontWeight:600}}>{n(mi.total_images)}</td></tr>
        </tbody>
      </table>
      <div style={{fontSize:12,color:"var(--muted)",marginTop:10}}>
        Every read submitted in <b>Analyze a case</b> is saved and folded into the model, so this number climbs (352,140 → 352,141 → …) and the model keeps improving. Updates automatically every few seconds.
      </div>
    </div>
  </>;
}

/* =================== Shell =================== */
function App() {
  const [mode, setMode] = useState("analyze");
  const Tab = ({id, label}) => <button className={"tab"+(mode===id?" active":"")} onClick={()=>setMode(id)}>{label}</button>;
  return <div className="wrap">
    <header>
      <div style={{display:"flex",alignItems:"center",gap:"12px"}}><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2NCA2NCIgd2lkdGg9IjY0IiBoZWlnaHQ9IjY0Ij4KICA8cmVjdCB4PSIyIiB5PSIyIiB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHJ4PSIxNSIgZmlsbD0iIzBmNzY2ZSIvPgogIDxjaXJjbGUgY3g9IjI4IiBjeT0iMjgiIHI9IjE1LjUiIGZpbGw9IiNlYWZhZjQiIHN0cm9rZT0iI2ZmZmZmZiIgc3Ryb2tlLXdpZHRoPSIzIi8+CiAgPGNpcmNsZSBjeD0iMjQiIGN5PSIyNSIgcj0iMy4yIiBmaWxsPSIjYzIzYTc2Ii8+CiAgPGNpcmNsZSBjeD0iMzIiIGN5PSIyMyIgcj0iMi42IiBmaWxsPSIjN2M1Y2Q2Ii8+CiAgPGNpcmNsZSBjeD0iMjYiIGN5PSIzMyIgcj0iMy42IiBmaWxsPSIjOWIzYWEwIi8+CiAgPGNpcmNsZSBjeD0iMzMiIGN5PSIzMiIgcj0iMi4zIiBmaWxsPSIjYzIzYTc2Ii8+CiAgPGxpbmUgeDE9IjM5IiB5MT0iMzkiIHgyPSI0OSIgeTI9IjQ5IiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iNS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiLz4KPC9zdmc+Cg==" width="42" height="42" alt="PathoAI logo" style={{borderRadius:"10px"}} /><h1 style={{margin:0}}>PathoAI · second-reader QA</h1></div>
      <div className="sub">An AI safety-net for the pathology sign-out. Quality assurance, not diagnosis.</div>
    </header>
    <div className="banner">
      <b>Assist, not replace.</b> PathoAI gives an independent second read and flags anything worth a second look. You decide the outcome; it never changes a sign-out.
    </div>
    <div className="tabs">
      <Tab id="analyze" label="Analyze a case"/>
      <Tab id="report" label="Report check"/>
      <Tab id="review" label="Review queue"/>
      <Tab id="model" label="Model"/>
    </div>
    {mode==="analyze" ? <GuidedFlow/> : mode==="report" ? <ReportCheck/> : mode==="review" ? <ReviewQueue/> : <ModelInfoView/>}
    <div className="foot">
      Every decision is appended to a local audit log (<span className="mono">audit_log.jsonl</span>).
      Data never leaves this machine. PathoAI outputs re-check recommendations only; it does not diagnose.
    </div>
  </div>;
}

ReactDOM.createRoot(document.getElementById("root")).render(<App/>);
