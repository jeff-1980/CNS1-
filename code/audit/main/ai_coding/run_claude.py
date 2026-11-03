"""Claude coder for AI static coding (RULES_v1 §1), same prompt and input packs as run_deepseek.py.

usage:
  python3 run_claude.py test NNN                 # one synchronous call, saved as run0
  python3 run_claude.py submit RUN NNN [NNN ...] # one Message Batch per run
  python3 run_claude.py collect RUN              # poll the run's batch, save each result
Key: AUDIT_ANTHROPIC_KEY. Current Claude models reject temperature, so effort is fixed instead.
No server-side model fallback: a refusal is recorded as a missing coding, not recoded by another model.
"""
import json, os, sys, time
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from run_deepseek import D, TAG, sha, templates

MODEL, EFFORT, MAX_TOKENS = 'claude-opus-5-5', 'high', 16000
client = anthropic.Anthropic(api_key=os.environ['AUDIT_ANTHROPIC_KEY'])
system, user_t, hashes = templates()


def params(n):
    paper = open(f'{D}/inputs/{n:03d}_paper.txt', encoding='utf-8').read()
    code = open(f'{D}/inputs/{n:03d}_code.txt', encoding='utf-8').read()
    user = user_t.replace('{PAPER}', paper).replace('{CODE}', code)
    return user, MessageCreateParamsNonStreaming(
        model=MODEL, max_tokens=MAX_TOKENS, system=system, output_config={'effort': EFFORT},
        messages=[{'role': 'user', 'content': user}])


def save(out, n, run, user, msg, extra):
    rec = {'draw_order': n, 'requested_model': MODEL, 'run': run, 'effort': EFFORT, **hashes, **extra,
           'system_sha256': sha(system), 'user_sha256': sha(user), 'response': msg.model_dump(mode='json')}
    json.dump(rec, open(f'{out}/{n:03d}.json', 'w'), ensure_ascii=False, indent=1)
    text = next((b.text for b in msg.content if b.type == 'text'), '')
    try:
        s = text[text.index('{'):text.rindex('}') + 1]; ok = len(json.loads(s))
    except ValueError:
        ok = 'PARSE_FAIL'
    u = msg.usage
    print(n, msg.model, msg.stop_reason, '| in', u.input_tokens, 'out', u.output_tokens, '| items', ok, flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'test':
        n = int(sys.argv[2]); out = f'{D}/raw/claude/{MODEL}{TAG}/run0'; os.makedirs(out, exist_ok=True)
        user, p = params(n); t0 = time.time()
        msg = client.messages.create(**p)
        save(out, n, 0, user, msg, {'called_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(t0)),
                                    'seconds': round(time.time() - t0, 1), 'batch': None})
    elif mode == 'submit':
        run, papers = int(sys.argv[2]), [int(a) for a in sys.argv[3:]]
        out = f'{D}/raw/claude/{MODEL}{TAG}/run{run}'; os.makedirs(out, exist_ok=True)
        b = client.messages.batches.create(requests=[Request(custom_id=f'{n:03d}', params=params(n)[1]) for n in papers])
        json.dump({'batch_id': b.id, 'run': run, 'papers': papers, 'submitted_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
                  open(f'{out}/_batch.json', 'w'), indent=1)
        print('run', run, 'batch', b.id, b.processing_status, len(papers), 'requests')
    elif mode == 'collect':
        run = int(sys.argv[2]); out = f'{D}/raw/claude/{MODEL}{TAG}/run{run}'
        meta = json.load(open(f'{out}/_batch.json'))
        while (b := client.messages.batches.retrieve(meta['batch_id'])).processing_status != 'ended':
            time.sleep(120)
        print('run', run, 'ended', b.request_counts)
        for r in client.messages.batches.results(meta['batch_id']):
            n = int(r.custom_id)
            if r.result.type == 'succeeded':
                save(out, n, run, params(n)[0], r.result.message, {'batch': meta['batch_id'], 'ended_at_utc': str(b.ended_at)})
            else:
                json.dump({'draw_order': n, 'run': run, 'batch': meta['batch_id'], 'result_type': r.result.type},
                          open(f'{out}/{n:03d}.failed.json', 'w'), indent=1)
                print(n, 'NOT SUCCEEDED:', r.result.type, flush=True)
