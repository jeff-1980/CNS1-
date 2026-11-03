"""RULES_v1 §2–§5: per-model codes, quote check, merge, agreement. No API calls.

usage: python3 analyze.py
Writes results/static_coding/ai_coding/{final_codes.csv, cells.csv, analysis.json}.
"""
import collections, csv, glob, json, os, re, unicodedata

ROOT = os.path.expanduser('~/audit_cns')
D = f'{ROOT}/results/static_coding/ai_coding'
# v1.1 recode: AI_TAG=_v1.1 AI_ITEMS=S4a_text_discloses,S8_traceability (outputs get the same suffix)
TAG = os.environ.get('AI_TAG', '')
MODELS = {'claude': f'raw/claude/claude-opus-5-5{TAG}', 'deepseek': f'raw/deepseek/deepseek-v4-pro{TAG}'}
RUNS = (1, 2, 3)
ITEMS = ['S1_split_unit', 'S2_validation', 'S3_checkpoint_rule_code', 'S4a_text_discloses', 'S4b_text_matches_code',
         'S5_runs', 'S5_spread_reported', 'S6_metrics', 'S7_noise_protocol', 'S8_traceability']
if os.environ.get('AI_ITEMS'):
    ITEMS = os.environ['AI_ITEMS'].split(',')
ALLOWED = {'S1_split_unit': set('ABCDU'), 'S2_validation': set('ABCU'), 'S3_checkpoint_rule_code': set('LVTEOQN'),
           'S4a_text_discloses': set('YPN'), 'S4b_text_matches_code': set('YDQN'), 'S5_spread_reported': set('YNU'),
           'S6_metrics': set('AMFB'), 'S7_noise_protocol': set('NTBOU')}
# "未说明"-type codes: an empty quote is valid evidence (RULES §3, §4)
UNSPEC = {i: {'U'} for i in ITEMS}
UNSPEC['S3_checkpoint_rule_code'] = {'N', 'Q'}
UNSPEC['S4b_text_matches_code'] = {'N', 'Q'}
UNSPEC['S4a_text_discloses'] = {'N'}
# Absence codes (something is not there) also need no quote. RULES_v1 §3 does not list them; set
# AI_ABSENCE_NO_QUOTE=0 for the literal reading (sensitivity analysis).
if os.environ.get('AI_ABSENCE_NO_QUOTE', '1') == '1':
    for _i, _c in (('S2_validation', 'A'), ('S5_spread_reported', 'N'), ('S7_noise_protocol', 'N'), ('S8_traceability', '0')):
        UNSPEC.setdefault(_i, {'U'}).add(_c)
DEFAULT = {'S1_split_unit': 'U', 'S2_validation': 'U', 'S7_noise_protocol': 'U', 'S3_checkpoint_rule_code': 'Q',
           'S4b_text_matches_code': 'Q', 'S4a_text_discloses': 'N', 'S5_runs': 'U', 'S5_spread_reported': 'U',
           'S6_metrics': None, 'S8_traceability': None}   # S6/S8 have no "未说明" code: flagged, left blank
# NFKC first: PDF text keeps ligatures (U+FB01 'fi'), which would otherwise be dropped
alnum = lambda s: re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKC', s).lower())


def norm_code(item, c):
    c = str(c).strip().upper().replace(' ', '')
    if item == 'S5_runs':
        m = re.fullmatch(r'(\d+)(\.0)?', c)
        return str(int(m.group(1))) if m else ('U' if c == 'U' else None)
    if item == 'S8_traceability':
        if c in ('0', 'O'):
            return '0'
        parts = set(c.split('+'))
        return '+'.join(p for p in 'CWGR' if p in parts) if parts <= set('CWGR') else None
    return c if c in ALLOWED[item] else None


def parse(model, path):
    r = json.load(open(path))['response']
    if model == 'claude':
        t = next((b['text'] for b in r['content'] if b['type'] == 'text'), '')
    else:
        t = r['choices'][0]['message']['content']
    return json.loads(t[t.index('{'):t.rindex('}') + 1], strict=False)   # strict=False: raw control chars in strings


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in ca) / n / n
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


if __name__ == '__main__':
    papers = sorted(int(os.path.basename(f)[:3]) for f in glob.glob(f'{D}/inputs/*_meta.json'))
    texts = {n: alnum(open(f'{D}/inputs/{n:03d}_paper.txt', encoding='utf-8').read() + '\n' +
                      open(f'{D}/inputs/{n:03d}_code.txt', encoding='utf-8').read()) for n in papers}
    raw = {}           # (model, n, run) -> {item: (code, quote)}
    invalid_codes = collections.Counter()
    for m, sub in MODELS.items():
        for n in papers:
            for r in RUNS:
                out = parse(m, f'{D}/{sub}/run{r}/{n:03d}.json')
                cell = {}
                for i in ITEMS:
                    v = out.get(i) or {}
                    c = norm_code(i, v.get('code', ''))
                    if c is None:
                        invalid_codes[(m, i)] += 1
                    cell[i] = (c, (v.get('quote') or '').strip())
                raw[m, n, r] = cell

    cells, final_rows = [], []
    stats = {'self_consistency': {}, 'quote_found_rate': {}, 'invalid_codes': {f'{m}:{i}': k for (m, i), k in invalid_codes.items()}}
    qtot = collections.Counter(); qok = collections.Counter()
    for (m, n, r), cell in raw.items():
        for i, (c, q) in cell.items():
            if q:
                qtot[m] += 1; qok[m] += texts[n].find(alnum(q)) >= 0 if alnum(q) else 0
    stats['quote_found_rate'] = {m: round(qok[m] / qtot[m], 4) for m in MODELS}
    stats['quotes_nonempty'] = dict(qtot)

    per_model = {}     # (m, n, i) -> (code or None if unstable, valid evidence, run codes)
    for m in MODELS:
        for i in ITEMS:
            all3 = two = 0
            for n in papers:
                codes = [raw[m, n, r][i][0] for r in RUNS]
                top, k = collections.Counter(codes).most_common(1)[0]
                all3 += k == 3; two += k >= 2
                if k >= 2 and top is not None:
                    quotes = [raw[m, n, r][i][1] for r in RUNS if raw[m, n, r][i][0] == top]
                    if top in UNSPEC[i] and not any(quotes):
                        valid = True
                    elif top in UNSPEC[i]:
                        valid = True   # RULES §3: a quote attached to an unspecified code is not checked
                    else:
                        valid = any(q and alnum(q) and texts[n].find(alnum(q)) >= 0 for q in quotes)
                    per_model[m, n, i] = (top, valid, codes)
                else:
                    per_model[m, n, i] = (None, False, codes)
            stats['self_consistency'][f'{m}:{i}'] = {'all3': round(all3 / len(papers), 3), 'ge2': round(two / len(papers), 3)}

    stats['kappa'] = {}; stats['merge'] = collections.Counter()
    for i in ITEMS:
        pairs = [(per_model['claude', n, i][0], per_model['deepseek', n, i][0]) for n in papers
                 if per_model['claude', n, i][0] is not None and per_model['deepseek', n, i][0] is not None]
        agree = sum(a == b for a, b in pairs) / len(pairs) if pairs else None
        k = max(2, len({c for p in pairs for c in p}))   # PABAK over the categories actually used
        stats['kappa'][i] = {'n': len(pairs), 'agreement': round(agree, 3) if agree is not None else None,
                             'kappa': None if i == 'S5_runs' else kappa(pairs),
                             'pabak': None if agree is None or i == 'S5_runs' else round((k * agree - 1) / (k - 1), 3)}

    for n in papers:
        row = {'draw_order': n}
        for i in ITEMS:
            (ca, va, ra), (cd, vd, rd) = per_model['claude', n, i], per_model['deepseek', n, i]
            if ca is not None and ca == cd:
                rule, final, options = ('agree_valid', ca, [ca]) if (va or vd) else ('agree_no_evidence', DEFAULT[i], [DEFAULT[i]])
            elif ca is not None and cd is not None and va != vd:
                rule, final = 'one_valid', ca if va else cd
                options = [final]
            elif ca is None or cd is None or (va and vd):
                rule, final = 'disputed', ''
                options = sorted({c for c in ([ca] if ca else ra) + ([cd] if cd else rd) if c is not None})
            else:
                rule, final, options = 'neither_valid', DEFAULT[i], [DEFAULT[i]]
            stats['merge'][rule] += 1
            row[i] = final if final is not None else ''
            row[i + '__options'] = '|'.join(o for o in options if o is not None)
            cells.append({'draw_order': n, 'item': i, 'claude': ca or 'UNSTABLE', 'claude_valid': va,
                          'claude_runs': '/'.join(str(c) for c in ra), 'deepseek': cd or 'UNSTABLE', 'deepseek_valid': vd,
                          'deepseek_runs': '/'.join(str(c) for c in rd), 'rule': rule, 'final': row[i], 'options': row[i + '__options']})
        final_rows.append(row)
    final_rows.insert(2, {**{k: '' for k in final_rows[0]}, 'draw_order': 3,
                          **({'S1_split_unit': 'X'} if 'S1_split_unit' in ITEMS else {})})   # 003: no full text
    final_rows.sort(key=lambda r: r['draw_order'])
    stats['merge'] = dict(stats['merge']); stats['n_cells'] = len(cells)

    # secondary outcome S3 (prereg §5): share of papers whose text does not state the model-selection rule (S4a = N)
    s4a = [] if 'S4a_text_discloses' not in ITEMS else [(r['S4a_text_discloses'], r['S4a_text_discloses__options'].split('|')) for r in final_rows if r['draw_order'] != 3]
    lo = sum(o == ['N'] for _, o in s4a); hi = sum('N' in o for _, o in s4a)
    if s4a:
        stats['S4a_not_disclosed'] = {'n_papers': len(s4a), 'lower': lo, 'upper': hi,
                                      'lower_pct': round(100 * lo / len(s4a), 1), 'upper_pct': round(100 * hi / len(s4a), 1)}
    stats['final_distribution'] = {i: dict(collections.Counter(r[i] or ('DISPUTED' if r[i + '__options'] else '')
                                                               for r in final_rows if r['draw_order'] != 3)) for i in ITEMS}

    with open(f'{D}/final_codes{TAG}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(final_rows[0].keys())); w.writeheader(); w.writerows(final_rows)
    with open(f'{D}/cells{TAG}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys())); w.writeheader(); w.writerows(cells)
    json.dump(stats, open(f'{D}/analysis{TAG}.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
