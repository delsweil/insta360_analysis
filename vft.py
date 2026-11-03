import sys, csv, collections
sys.path.insert(0,'.')
import numpy as np, suggest as S
A="/mnt/user-data/attachments/9fd2a633-998b-562b-b933-abd2243587a9"
games={"B-Jugend":(A+"/16efab02-1793559049067_ASN_Pfeil_Ph_nix_vs_Rot_Weiss_b-Jugend__frames.csv",A+"/23cb9da7-1793559049067_ASN_Pfeil_Ph_nix_vs_Rot_Weiss_b-Jugend__events.csv"),
       "A-Jugend":(A+"/0390c60a-1793549539849_ASN_Pfeil_Ph_nix_vs_Eibach_A-Jugend__frames.csv",A+"/5a74f531-1793549539849_ASN_Pfeil_Ph_nix_vs_Eibach_A-Jugend__events_1.csv")}
TEAMS=sys.argv[1]
W,TOL,LEAD=30.0,10.0,12.0
data={}
for g,(fr,ev) in games.items():
    tab=S.table_from_csv(fr); sg=S.suggest(tab,teams=TEAMS)
    t=np.asarray(tab["t"],float)
    ev=[(float(r['video_t'])+LEAD,r['label']) for r in csv.DictReader(open(ev))]
    data[g]=(t,[s['t_peak'] for s in sg],ev,sg)
    print(g,"duration",round(t.max()/60,1),"min; suggestions",len(sg),collections.Counter(s['source'] for s in sg))
classes=["shot_home","goal_home","setpiece_home","shot_away","setpiece_away","tactical"]
def table(sel):
    tp=fp=fn=tn=0
    for g,(t,sp,ev,_) in data.items():
        evt=[e for e,l in ev if sel(l)]
        for a in np.arange(t.min(),t.max(),W):
            if not ((t>=a)&(t<a+W)).any(): continue
            act=any(a<=e<a+W for e in evt)
            pred=any(a-TOL<=p<a+W+TOL for p in sp)
            if act and pred: tp+=1
            elif act: fn+=1
            elif pred: fp+=1
            else: tn+=1
    return tp,fp,fn,tn
rows=[]
for name,sel in [(c,(lambda l,c=c:l==c)) for c in classes]+[("shot+goal home",lambda l:l in("shot_home","goal_home")),("any event",lambda l:True)]:
    tp,fp,fn,tn=table(sel); n=tp+fp+fn+tn
    print(f"{name:15} TP{tp:4} FP{fp:4} FN{fn:4} TN{tn:4}  windows {n}  sens {tp/max(1,tp+fn):.2f}  prec {tp/max(1,tp+fp):.2f}  spec {tn/max(1,tn+fp):.2f}")
print("--- per game, shot+goal home")
for g in data:
    d={g:data[g]}; old=dict(data); data.clear(); data.update(d)
    print(g,table(lambda l:l in("shot_home","goal_home")))
    data.clear(); data.update(old)
