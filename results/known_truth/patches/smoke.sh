#!/bin/bash
# known-truth smoke: 1 seed, 2-3 epochs per object, separate record dir
R=~/audit_ext/known_truth/smoke; rm -rf $R; mkdir -p $R
cd ~/audit_ext
./kt58.sh $R smoke --groups CDAN-mixup --seeds 0-0 --epochs 2 | tail -3
./kt54.sh $R smoke --groups TGST,Random-text --repeats 0-0 --epochs 2 | tail -3
./kt76.sh $R smoke --repeats 0-0 --epochs 3 | tail -3
./kt37.sh $R smoke --repeats 0-0 --rounds 2 | tail -3
./kt62.sh $R smoke --seeds 0-0 --epochs 2 | tail -3
./kt53.sh $R smoke --groups Backbone_CNN --seeds 0-0 --max_epoch 2 | tail -3
python3 - <<'X'
import json, glob, os
for f in sorted(glob.glob(os.path.expanduser('~/audit_ext/known_truth/smoke/*/*/seed*.json'))):
    d = json.load(open(f)); e = d['epochs']
    last = e[max(e, key=int)]
    full = d.get('kt_full', {}).get(max(e, key=int))
    print(f.split('smoke/')[1], 'epochs', len(e), {k: round(v['acc'], 4) for k, v in last.items()}, 'kt_full', full, 'check', (d.get('author_check') or {}).get('matches'))
X
