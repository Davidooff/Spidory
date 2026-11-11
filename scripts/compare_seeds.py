"""Collect every model x seed into one table (re-scored numbers for old runs) and print means."""
import json, glob, os, statistics as st, sys
R = '/home/david/projects/Spidory/runs'
rows = {}
def add(model, seed, path):
    d = json.load(open(path)); bk = d['by_k']
    k = '3' if '3' in bk else '1'
    e = dict(bk[k]); e['k'] = int(k); e['params'] = d['params']
    e['k1'] = bk['1']['mean_loss']
    if d.get('learned'): e['neuron'] = d['learned']
    rows[(model, seed)] = e
for p in glob.glob(f'{R}/remote_r6_deep/rescore_*_s*.json'):
    n = os.path.basename(p)[8:-5]; m, s = n.rsplit('_s', 1); add(m, int(s), p)
for p in glob.glob(f'{R}/remote_r6_*/results_*_s*.json'):
    n = os.path.basename(p)[8:-5]; m, s = n.rsplit('_s', 1); add(m, int(s), p)
order = ['noloop', 'noloop_deep', 'plain', 'plain_signal', 'plain_signal_nonorm', 'perloop', 'token', 'token_signal', 'token_signal_once', 'between_signal']
cols = [('mean_loss', 'loss', 1, 3), ('maths_accuracy', 'maths', 100, 1), ('explain_accuracy', 'expl', 100, 1),
        ('compose_accuracy', 'comp', 100, 1), ('mixed_accuracy', 'mixed', 100, 1), ('gsm8k_accuracy', 'gsm', 100, 1)]
out = {}
print('model | k | seeds | ' + ' | '.join(c[1] + ' (s0/s1/s2) mean' for c in cols) + ' | k1 loss mean')
for m in order:
    ss = sorted(s for (mm, s) in rows if mm == m)
    if not ss: continue
    line = [m, str(rows[(m, ss[0])]['k']), ','.join(map(str, ss))]
    out[m] = {'seeds': ss, 'params': rows[(m, ss[0])]['params']}
    for key, name, mul, nd in cols:
        v = [rows[(m, s)][key] * mul for s in ss]
        out[m][name] = v
        sd = f' ±{st.stdev(v):.{nd}f}' if len(v) > 1 else ''
        line.append('/'.join(f'{x:.{nd}f}' for x in v) + f'  {st.mean(v):.{nd}f}{sd}')
    k1 = [rows[(m, s)]['k1'] for s in ss]; out[m]['k1'] = k1
    line.append(f'{st.mean(k1):.3f}')
    if all('neuron' in rows[(m, s)] for s in ss):
        nl = [rows[(m, s)]['neuron']['mean_loss'] for s in ss]; nr = [rows[(m, s)]['neuron'].get('mean_runs') for s in ss]
        na = [rows[(m, s)]['neuron']['maths_accuracy'] * 100 for s in ss]
        out[m]['neuron'] = {'loss': nl, 'runs': nr, 'maths': na}
        line.append(f'neuron loss {st.mean(nl):.3f} maths {st.mean(na):.1f} runs {st.mean([r for r in nr if r]):.2f}')
    print(' | '.join(line))
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else '/dev/null', 'w'), indent=1)
