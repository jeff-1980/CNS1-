"""Build the known-truth drivers (audit_ktNN.py) and run scripts (ktNN.sh) from the main-audit ones.
Evaluation-only changes; every replacement must match exactly once. Run once in WSL; re-running is a no-op."""
import glob, os, re

EXT = os.path.expanduser('~/audit_ext')
R = lambda o: glob.glob(f'{EXT}/main_objects/{o}_*')[0]
IMP = "from audit.main.kt_split import Splitter, log_halves, acc as kt_acc, balanced as kt_bal  # [audit kt]\n"


def edit(path, pairs, out=None):
    t = open(path, encoding='utf-8').read()
    if out is None and '[audit kt]' in t:
        return print('already done:', path)
    for a, b in pairs:
        n = t.count(a)
        assert n == 1, (path, n, a[:80])
        t = t.replace(a, b)
    open(out or path, 'w', encoding='utf-8').write(t)
    print('wrote', out or path)


def driver(o, pairs):
    src = f'{R(o)}/audit_run{o}.py'
    edit(src, [("from audit.main.epoch_record import Recorder\n", "from audit.main.epoch_record import Recorder\n" + IMP)] + pairs,
         out=f'{R(o)}/audit_kt{o}.py')
    sh = open(f'{EXT}/run{o}.sh').read()
    assert sh.count(f'audit_run{o}.py') == 1 and sh.count(f'logs{o}') >= 1
    sh = sh.replace(f'audit_run{o}.py', f'audit_kt{o}.py').replace(f'logs{o}', f'logs_kt{o}')
    open(f'{EXT}/kt{o}.sh', 'w').write(sh); os.chmod(f'{EXT}/kt{o}.sh', 0o755)


# 58: their hook already receives the full test predictions in fixed order; metric = balanced accuracy (4 classes)
driver('58', [("""        def hook(ep, loss, acc, pred, y):
            rec.log(ep, "test", balanced(pred, y), float(loss))""",
               """        sp = Splitter()

        def hook(ep, loss, acc, pred, y):
            log_halves(rec, ep, pred, y, sp, metric=lambda p, t, m=None: kt_bal(p, t, m, range(4)), loss=float(loss))
            assert abs(rec.d["epochs"][str(ep)]["test"]["acc"] - balanced(pred, y)) < 1e-9""")])

# 54: their evaluate_model already returns predictions and labels; keep them instead of discarding
d = R('54') + '/tgst_project/scripts/'
edit(d + 'experiment_v2.py', [
    ("AUDIT_HOOK = globals().get('AUDIT_HOOK'); AUDIT_FINAL = globals().get('AUDIT_FINAL')",
     "AUDIT_HOOK = globals().get('AUDIT_HOOK'); AUDIT_FINAL = globals().get('AUDIT_FINAL'); AUDIT_PRED = globals().get('AUDIT_PRED')  # [audit kt]"),
    ("            test_acc, _, _, _ = evaluate_model(model, test_loader, device, is_tgst)\n            if AUDIT_HOOK: AUDIT_HOOK(epoch, test_acc, model, test_loader)  # [audit]\n",
     "            test_acc, _kt_p, _kt_l, _ = evaluate_model(model, test_loader, device, is_tgst)\n            if AUDIT_HOOK: AUDIT_HOOK(epoch, test_acc, model, test_loader)  # [audit]\n            if AUDIT_PRED: AUDIT_PRED(epoch, _kt_p, _kt_l)  # [audit kt]\n")])
t = open(d + 'experiment_ablation.py', encoding='utf-8').read()
m = re.search(r"( *)test_acc, _, _, _ = evaluate_model\(model, test_loader, device\)\n( *)if AUDIT_HOOK: AUDIT_HOOK\(epoch, test_acc, model, test_loader\)  # \[audit\]\n", t)
if '[audit kt]' not in t:
    assert m and len(re.findall(r"test_acc, _, _, _ = evaluate_model\(model, test_loader, device\)\n", t)) == 1
    edit(d + 'experiment_ablation.py', [
        ("AUDIT_HOOK = globals().get('AUDIT_HOOK'); AUDIT_FINAL = globals().get('AUDIT_FINAL')",
         "AUDIT_HOOK = globals().get('AUDIT_HOOK'); AUDIT_FINAL = globals().get('AUDIT_FINAL'); AUDIT_PRED = globals().get('AUDIT_PRED')  # [audit kt]"),
        (m.group(0), f"{m.group(1)}test_acc, _kt_p, _kt_l, _ = evaluate_model(model, test_loader, device)\n"
                     f"{m.group(2)}if AUDIT_HOOK: AUDIT_HOOK(epoch, test_acc, model, test_loader)  # [audit]\n"
                     f"{m.group(2)}if AUDIT_PRED: AUDIT_PRED(epoch, _kt_p, _kt_l)  # [audit kt]\n")])
driver('54', [
    ("        final = {}\n",
     "        sp = Splitter()\n\n"
     "        def pred_hook(ep, p, l):\n"
     "            S, T = sp.masks('test', l)\n"
     "            rec.log(ep, 'S', kt_acc(p, l, S)); rec.log(ep, 'T', kt_acc(p, l, T))\n"
     "            rec.d.setdefault('kt_full', {})[str(ep)] = kt_acc(p, l)\n\n"
     "        final = {}\n"),
    ("init_globals=dict(AUDIT_HOOK=hook, AUDIT_FINAL=lambda v: final.update(v=float(v))))",
     "init_globals=dict(AUDIT_HOOK=hook, AUDIT_FINAL=lambda v: final.update(v=float(v)), AUDIT_PRED=pred_hook))")])

# 76 / 53: collect the predictions their own eval loop computes (no extra inference)
for o, f, phase in (('76', '/DAGCN/utils/train_utils_combines.py', 'target_val'), ('53', '/utils/train_utils.py', 'val')):
    p = R(o) + f; t = open(p, encoding='utf-8').read()
    if '[audit kt]' not in t:
        m = re.search(r"\n( *)pred = logits\.argmax\(dim=1\)\n", t); assert m and t.count("pred = logits.argmax(dim=1)\n") == 1
        top = re.search(r"\nAUDIT_HOOK = None[^\n]*\n", t); assert top
        edit(p, [(m.group(0), m.group(0) + f"{m.group(1)}if AUDIT_PRED and phase == '{phase}': AUDIT_PRED(pred, labels)  # [audit kt]\n"),
                 (top.group(0), top.group(0) + "AUDIT_PRED = None  # [audit kt] known-truth test, see audit_kt*.py\n")])

driver('76', [
    ("    final = {}\n",
     "    final = {}\n    sp = Splitter(); buf = []\n    TU.AUDIT_PRED = lambda p, l: buf.append((p.cpu().numpy(), l.cpu().numpy()))\n"),
    ("        rec.log(ep, \"test\" if phase == \"target_val\" else \"source_val\", float(acc), float(loss))\n",
     "        rec.log(ep, \"test\" if phase == \"target_val\" else \"source_val\", float(acc), float(loss))\n"
     "        if phase == \"target_val\":\n"
     "            import numpy as np\n"
     "            p = np.concatenate([b[0] for b in buf]); l = np.concatenate([b[1] for b in buf]); buf.clear()\n"
     "            S, T = sp.masks('test', l)\n"
     "            rec.log(ep, 'S', kt_acc(p, l, S)); rec.log(ep, 'T', kt_acc(p, l, T))\n"
     "            rec.d.setdefault('kt_full', {})[str(ep)] = kt_acc(p, l)\n")])

driver('53', [
    ("        st = {\"best\": 0.0, \"accs\": []}\n",
     "        st = {\"best\": 0.0, \"accs\": []}\n        sp = Splitter(); buf = []\n"
     "        TU.AUDIT_PRED = lambda p, l: buf.append((p.cpu().numpy(), l.cpu().numpy()))\n"),
    ("                rec.log(ep, \"test\", float(acc), float(loss))\n",
     "                rec.log(ep, \"test\", float(acc), float(loss))\n"
     "                p = np.concatenate([b[0] for b in buf]); l = np.concatenate([b[1] for b in buf]); buf.clear()\n"
     "                S, T = sp.masks('test', l)\n"
     "                rec.log(ep, 'S', kt_acc(p, l, S)); rec.log(ep, 'T', kt_acc(p, l, T))\n"
     "                rec.d.setdefault('kt_full', {})[str(ep)] = kt_acc(p, l)\n"),
    ("            if phase == \"val\":\n", "            if phase != \"val\":\n                buf.clear()   # [audit kt] only val predictions\n            if phase == \"val\":\n")])

# 37: one extra isolated pass inside their hook, same inter-domain definition as their evaluate()
driver('37', [
    ("    st = {\"best\": -1.0}\n", "    st = {\"best\": -1.0}\n    sp = Splitter()\n"),
    ("            loss = inter_ce(model, test_loaders)\n",
     "            loss = inter_ce(model, test_loaders)\n"
     "            kt = {'S': [], 'T': [], 'full': []}\n"
     "            n_dom = len(test_loaders)\n"
     "            for i, net in enumerate(model.nets_list):\n"
     "                per = {'S': [], 'T': [], 'full': []}\n"
     "                for j, dl in enumerate(test_loaders):\n"
     "                    if i % n_dom == j: continue\n"
     "                    P, L = [], []\n"
     "                    with torch.no_grad():\n"
     "                        for x, y, _ in dl:\n"
     "                            P.append(net(x.to(model.device)).argmax(1).cpu().numpy()); L.append(y.numpy())\n"
     "                    P, L = np.concatenate(P), np.concatenate(L); S, T = sp.masks(f'dom{j}', L)\n"
     "                    per['S'].append(kt_acc(P, L, S)); per['T'].append(kt_acc(P, L, T)); per['full'].append(kt_acc(P, L))\n"
     "                for k in kt: kt[k].append(float(np.mean(per[k])))\n"),
    ("        rec.log(ep, \"test\", acc, loss)\n",
     "        rec.log(ep, \"test\", acc, loss)\n"
     "        rec.log(ep, 'S', float(np.mean(kt['S']))); rec.log(ep, 'T', float(np.mean(kt['T'])))\n"
     "        rec.d.setdefault('kt_full', {})[str(ep)] = float(np.mean(kt['full']))\n")])

# 62: their 10 test re-samplings; split each one, average over the 10 as for the full value
driver('62', [
    ("    accs, ces, f1s = [], [], []\n    for X, Y in TESTS:\n",
     "    accs, ces, f1s, aS, aT = [], [], [], [], []\n    for k, (X, Y) in enumerate(TESTS):\n"),
    ("        f1s.append(f1_score(Y.numpy(), pred.numpy(), average=\"macro\"))\n    return float(np.mean(accs)), float(np.mean(ces)), float(np.mean(f1s))\n",
     "        f1s.append(f1_score(Y.numpy(), pred.numpy(), average=\"macro\"))\n"
     "        S, T = SP.masks(k, Y.numpy()); aS.append(kt_acc(pred.numpy(), Y.numpy(), S)); aT.append(kt_acc(pred.numpy(), Y.numpy(), T))\n"
     "    return float(np.mean(accs)), float(np.mean(ces)), float(np.mean(f1s)), float(np.mean(aS)), float(np.mean(aT))\n"),
    ("TESTS = test_sets()\n", "TESTS = test_sets()\nSP = Splitter()\n"),
    ("                model.eval(); acc, ce, f1 = eval_tests(model)\n", "                model.eval(); acc, ce, f1, accS, accT = eval_tests(model)\n"),
    ("            rec.log(ep, \"test\", acc, ce)\n", "            rec.log(ep, \"test\", acc, ce)\n            rec.log(ep, 'S', accS); rec.log(ep, 'T', accT)\n")])
