import json,sys,glob,os
E={'data.type':{'lab_experiment','fabrication','field_operational','public_dataset','simulation','theory','survey','mixed'},
'data.public':{'yes','no','unclear'},'data.real_world':{'yes','no'},'pure_computational':{'yes','no'},'requires_hardware':{'yes','no'},
'claim_level':{'T1','T2','T3','T4'},'lever':{'none','meta_research','large_scale_operational_data','model_plus_real_event','pure_method_theory','new_dataset'},
'mech_relevance':{'core','adjacent','tagged_only'}}
EV={'fabrication','wet_lab','bench_experiment','field_trial','simulation','theory','data_analysis'}
bad=0
for f in sorted(glob.glob('cards/out/batch_*.jsonl')):
    b=os.path.basename(f); inp=[json.loads(l)['id'] for l in open('cards/in/'+b)]
    rows=[json.loads(l) for l in open(f)]
    ids=[r.get('id') for r in rows]
    if ids!=inp: print(b,'ID ORDER/COUNT MISMATCH',len(ids),len(inp)); bad+=1
    for r in rows:
        for k,s in E.items():
            v=r.get(k.split('.')[0]) if '.' not in k else (r.get('data') or {}).get(k.split('.')[1])
            if v not in s: print(b,r.get('id'),k,repr(v)); bad+=1
        if not set(r.get('evidence') or [])<=EV: print(b,r.get('id'),'evidence',r.get('evidence')); bad+=1
print('files',len(glob.glob('cards/out/batch_*.jsonl')),'bad',bad)
