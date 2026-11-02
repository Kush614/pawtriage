"""Generate the thin notebooks (logic lives in src/)."""
import nbformat as nbf
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks"
SETUP = """import sys, json, os
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
from dotenv import load_dotenv; load_dotenv(ROOT / '.env')
import nest_asyncio; nest_asyncio.apply()
import asyncio"""

books = {
"00_rubric_check": ["# 00 · Rubric + vocab check\nValidates `data/rubric.csv` against the fixed vocabularies and prints coverage. Cost: $0, <5s.",
 SETUP, """import pt_data as D, pt_prompt as P
rubric = D.load_rubric()
errs = D.validate_rubric(rubric)
print('ERRORS:', errs or 'none')
print(json.dumps(D.coverage(rubric), indent=1))
print('conditions:', len(P.vocab()['conditions']), '| image findings:', P.vocab()['image_findings'])
assert not errs"""],
"01_generate_and_split": ["# 01 · Generate owner messages, split by scenario, LOCK\nLabels come from the rubric, never the generator. Cost ≈ $2–6 (OpenRouter), ~5 min.",
 SETUP, """import pt_data as D, pt_pipeline as PL
from pt_llm import LLM
llm = LLM(concurrency=16, cache_name='generate')
text_rows = await PL.generate_text_rows(llm, per_scenario=40)
print(len(text_rows), 'text rows')""",
 """img_train, img_test = await PL.build_image_rows(llm)
print(len(img_train), 'image train rows;', len(img_test), 'image test rows')
print(open(ROOT/'data/generated/image_split_log.json').read())""",
 """text_train, text_test = PL.split_text(text_rows)
PL.write_candidates(text_train, text_test, img_train, img_test)
import pandas as pd
print(len(text_train), 'train /', len(text_test), 'test candidates. Train files are final -> 04_sft can launch now.')
print('Human check: two teammates fill `action` in data/splits/review_test.csv with drop / fix_label:<urgency>')
pd.read_csv(ROOT/'data/splits/review_test.csv').head(40)""",
 """# After review: apply + LOCK (refuses if already locked)
print(PL.apply_review_and_lock())
print(open(ROOT/'data/splits/LOCKED.md').read())"""],
"02_baseline": ["# 02 · Baseline: GLM 5.3 Flash serverless, temp 0\nCost ≈ $0.30, ~3 min.",
 SETUP, """import pt_systems as S, pt_score as SC
from pt_llm import LLM
llm = LLM(concurrency=16, cache_name='score')
rows = S.test_rows()
res = await S.base(llm, rows)
print(json.dumps(SC.by_modality(res), indent=1))"""],
"03_harness": ["# 03 · Harness: base + full rubric in the system prompt\nCost ≈ $1–2 (long prompt), ~4 min.",
 SETUP, """import pt_systems as S, pt_score as SC
from pt_llm import LLM
llm = LLM(concurrency=16, cache_name='score')
res = await S.harness(llm, S.test_rows())
print(json.dumps(SC.by_modality(res), indent=1))"""],
"04_sft": ["# 04 · Vision LoRA SFT (Fireworks serverless training)\nRuns `src/pt_sft.py` in the cookbook venv. Wall time ≈ 1–1.5 h. Launch by 12:45.",
 SETUP, """import pt_sft
info = pt_sft.build()
print(json.dumps(info, indent=1))""",
 """meta = pt_sft.launch()   # background process; safe to close the notebook
meta""",
 """# Re-run to see progress
for l in pt_sft.sync_train_log()[-10:]: print(l)"""],
"05_score_all": ["# 05 · Score every system on the locked test set\nAll calls cached in results/cache, so reruns are free. Requires FIREWORKS_DEPLOYMENT_MODEL.",
 SETUP, """import pt_systems as S, pt_score as SC
from pt_llm import LLM
llm = LLM(concurrency=16, cache_name='score')
rows = S.test_rows()
out = {}
for name, fn in [('base', S.base), ('harness_rubric', S.harness), ('sft', S.sft), ('sft_safety', S.sft_safety),
                 ('sft_router', S.sft_router), ('frontier', S.frontier)]:
    try:
        out[name] = await fn(llm, rows)
        print(name, json.dumps(SC.by_modality(out[name])['all']))
    except Exception as e:
        print(name, 'FAILED:', e)
# acceptance: identical prompt per run
print({n: {r['prompt_hash'] for r in v} for n, v in out.items()})"""],
"06_report": ["# 06 · McNemar + cluster bootstrap + charts + summary\nCost: $0.",
 SETUP, """import pt_report as R, pt_score as SC
names = ['base', 'harness_rubric', 'frontier', 'sft', 'sft_safety', 'sft_router']
systems = {n: SC.load_results(n) for n in names if SC.load_results(n)}
summary = R.compare_all(systems)
R.write_summary(summary)
print(R.charts(summary))
for mod in ('text', 'image', 'all'):
    print(f'\\n### {mod}\\n' + R.markdown_table(summary, mod))""",
 """print('REAL GAINS (p<0.05 AND CI excludes 0):')
for n, e in summary.items():
    for mod, res in e['vs_base'].items():
        for f, v in res.items():
            if isinstance(v, dict) and v.get('real_gain'):
                print(f'  {n} · {mod} · {f}: {v[\"bootstrap\"][\"mean_gap\"]*100:+.1f}pp, p={v[\"mcnemar\"][\"p_exact\"]:.2g}')
if 'video' in next(iter(summary.values()))['metrics']:
    print('video (n small, no significance claim):', {n: e['metrics']['video'] for n, e in summary.items()})""",
 """from IPython.display import Image, display
display(Image(str(ROOT/'results/charts/mcnemar.png'))); display(Image(str(ROOT/'results/charts/bootstrap.png')))"""],
}
for name, cells in books.items():
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(cells[0])] + [nbf.v4.new_code_cell(c) for c in cells[1:]]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB / f"{name}.ipynb")
print("wrote", len(books))
