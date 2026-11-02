import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pt_systems as S, pt_score as SC
from pt_llm import LLM
async def main(names):
    llm = LLM(concurrency=16, cache_name="score")
    rows = S.test_rows()
    for n in names:
        fn = {"base": S.base, "harness_rubric": S.harness, "frontier": S.frontier, "sft": S.sft,
              "sft_safety": S.sft_safety, "sft_router": S.sft_router}[n]
        try:
            res = await fn(llm, rows)
            print(n, json.dumps({k: v for k, v in SC.by_modality(res).items() if v.get("n")}), flush=True)
        except Exception as e:
            print(n, "FAILED", repr(e)[:500], flush=True)
asyncio.run(main(sys.argv[1:]))
