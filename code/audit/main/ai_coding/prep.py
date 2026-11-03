"""Build the per-paper input pack for AI static coding (RULES_v1 §1, PROMPT_v1 "输入包").

usage: python3 prep.py NNN [NNN ...]
Writes results/static_coding/ai_coding/inputs/NNN_paper.txt, NNN_code.txt, NNN_meta.json.
Repos are shallow-cloned to /mnt/d/audit_data/ai_coding_repos/NNN (not committed).
"""
import hashlib, json, os, re, subprocess, sys
from pypdf import PdfReader

ROOT = os.path.expanduser('~/audit_cns')
PDF = '/mnt/d/coder2/pdf'
OUT = f'{ROOT}/results/static_coding/ai_coding/inputs'
REPOS = '/mnt/d/audit_data/ai_coding_repos'
KEEP = ('.py', '.ipynb', '.m', '.yaml', '.yml', '.json', '.cfg', '.txt', '.md')
CODE_TOKEN_CAP = 60000          # rough: 1 token ~ 4 chars
LINK = re.compile(r'https?://(?:www\.)?(?:github\.com|gitlab\.com|gitee\.com)/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+', re.I)

sha = lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()


def paper_text(n):
    rd = PdfReader(f'{PDF}/{n:03d}.pdf')
    return '\n'.join(f'[[PAGE {i + 1}]]\n{p.extract_text() or ""}' for i, p in enumerate(rd.pages)), len(rd.pages)


def find_links(text):
    # PDF extraction can break URLs across lines; join "github.com/x/\ny" first
    joined = re.sub(r'(github\.com|gitlab\.com|gitee\.com)/\s*([^\s/]+)/\s*', r'\1/\2/', text, flags=re.I)
    # ...and after a hyphen at a line end ("SQ-\ndataset")
    joined = re.sub(r'((?:github\.com|gitlab\.com|gitee\.com)/[^\s/]+/[^\s]*-)\n', r'\1', joined, flags=re.I)
    seen = []
    for u in LINK.findall(joined):
        u = u.rstrip('.,;)').removesuffix('.git')
        if u.lower() not in [s.lower() for s in seen]:
            seen.append(u)
    return seen


def code_bundle(n, links):
    if not links:
        return 'NO_CODE', {'status': 'no_link'}
    notes = []
    for url in links:
        dst = f'{REPOS}/{n:03d}/{url.rstrip("/").split("/")[-1]}'
        if not os.path.exists(dst):
            r = subprocess.run(['git', 'clone', '-q', '--depth', '1', url, dst], capture_output=True, text=True,
                               timeout=600, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
            if r.returncode:
                notes.append({'url': url, 'status': 'clone_failed', 'err': r.stderr.strip()[-200:]})
                continue
        commit = subprocess.run(['git', '-C', dst, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
        notes.append({'url': url, 'status': 'ok', 'commit': commit, 'dir': dst})
    ok = [x for x in notes if x['status'] == 'ok']
    if not ok:
        return 'NO_CODE', {'status': 'all_clones_failed', 'repos': notes}
    files = []
    for x in ok:
        for d, _, fs in os.walk(x['dir']):
            if '/.git' in d:
                continue
            for f in fs:
                if f.lower().endswith(KEEP):
                    p = os.path.join(d, f)
                    rel = os.path.relpath(p, f'{REPOS}/{n:03d}')
                    files.append((0 if f.lower().startswith('readme') else 1, rel.lower(), rel, p))
    parts, total, truncated = [], 0, False
    for _, _, rel, p in sorted(files):
        try:
            lines = open(p, encoding='utf-8', errors='replace').read().splitlines()
        except OSError:
            continue
        block = f'[[FILE {rel}]]\n' + '\n'.join(f'{i + 1}: {l}' for i, l in enumerate(lines))
        if (total + len(block)) / 4 > CODE_TOKEN_CAP:
            truncated = True
            break
        parts.append(block); total += len(block)
    text = '\n\n'.join(parts) + ('\n\n[[已截断]]' if truncated else '')
    return text, {'status': 'ok', 'repos': notes, 'n_files_kept': len(parts), 'n_files_total': len(files), 'truncated': truncated}


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for a in sys.argv[1:]:
        n = int(a)
        paper, pages = paper_text(n)
        links = find_links(paper)
        code, cmeta = code_bundle(n, links)
        open(f'{OUT}/{n:03d}_paper.txt', 'w', encoding='utf-8').write(paper)
        open(f'{OUT}/{n:03d}_code.txt', 'w', encoding='utf-8').write(code)
        meta = {'draw_order': n, 'pages': pages, 'links_found': links, 'code': cmeta,
                'paper_sha256': sha(paper), 'code_sha256': sha(code),
                'paper_chars': len(paper), 'code_chars': len(code)}
        json.dump(meta, open(f'{OUT}/{n:03d}_meta.json', 'w'), ensure_ascii=False, indent=1)
        print(n, pages, 'pages', len(paper), 'chars | links', links, '| code', cmeta.get('status'), len(code), 'chars')
