"""Call DeepSeek on prepared input packs with PROMPT_v1 (RULES_v1 §1).

usage: python3 run_deepseek.py MODEL RUN NNN [NNN ...]
Reads DEEPSEEK_API_KEY from the environment. Writes raw request metadata and the full
response to results/static_coding/ai_coding/raw/deepseek/<MODEL>/run<RUN>/NNN.json.
"""
import hashlib, json, os, re, sys, time, urllib.request

ROOT = os.path.expanduser('~/audit_cns')
D = f'{ROOT}/results/static_coding/ai_coding'
sha = lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()


def templates():
    t = open(f'{D}/PROMPT_v1.md', encoding='utf-8').read()
    system = t.split('## 系统提示', 1)[1].split('## 用户消息', 1)[0].strip()
    user = t.split('## 用户消息', 1)[1].split('## 输入包', 1)[0].strip()
    cb = open(f'{ROOT}/results/static_coding/CODEBOOK_v1.md', encoding='utf-8').read()
    # only the item/category definitions; the "怎么编" section addresses human coders
    items = '## 条目与类别' + cb.split('## 条目与类别', 1)[1].split('## 一致性', 1)[0].rstrip()
    return system.replace('{CODEBOOK}', items), user, {'prompt_sha256': sha(t), 'codebook_sha256': sha(cb)}


def call(model, system, user):
    body = {'model': model, 'temperature': 0, 'response_format': {'type': 'json_object'},
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
    req = urllib.request.Request('https://api.deepseek.com/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json',
                                          'Authorization': 'Bearer ' + os.environ['DEEPSEEK_API_KEY']})
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=900))
        except Exception as e:  # retry transient errors only
            if attempt == 3:
                raise
            print('  retry after', repr(e)[:120]); time.sleep(2 ** (attempt + 2))


if __name__ == '__main__':
    model, run, papers = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    system, user_t, hashes = templates()
    out = f'{D}/raw/deepseek/{model}/run{run}'
    os.makedirs(out, exist_ok=True)
    for a in papers:
        n = int(a)
        if os.path.exists(f'{out}/{n:03d}.json'):
            print(n, 'exists, skipped'); continue
        paper = open(f'{D}/inputs/{n:03d}_paper.txt', encoding='utf-8').read()
        code = open(f'{D}/inputs/{n:03d}_code.txt', encoding='utf-8').read()
        user = user_t.replace('{PAPER}', paper).replace('{CODE}', code)
        t0 = time.time()
        r = call(model, system, user)
        rec = {'draw_order': n, 'requested_model': model, 'run': run, 'temperature': 0,
               'called_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0)),
               'seconds': round(time.time() - t0, 1), **hashes,
               'system_sha256': sha(system), 'user_sha256': sha(user), 'response': r}
        json.dump(rec, open(f'{out}/{n:03d}.json', 'w'), ensure_ascii=False, indent=1)
        u = r.get('usage', {})
        try:
            parsed = json.loads(r['choices'][0]['message']['content']); ok = len(parsed)
        except Exception:
            ok = 'PARSE_FAIL'
        print(n, r.get('model'), '| tokens in', u.get('prompt_tokens'), 'out', u.get('completion_tokens'),
              '| items', ok, '|', rec['seconds'], 's')
