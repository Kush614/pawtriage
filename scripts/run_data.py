"""One-shot: generate text + image rows, split, write train + test candidates. Then launch SFT (--train)."""
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pt_pipeline as PL
from pt_llm import LLM


async def main():
    llm = LLM(concurrency=16, cache_name="generate")
    text_rows = await PL.generate_text_rows(llm, per_scenario=40)
    print("text rows", len(text_rows), flush=True)
    img_train, img_test = await PL.build_image_rows(llm)
    print("image rows", len(img_train), len(img_test), flush=True)
    text_train, text_test = PL.split_text(text_rows)
    PL.write_candidates(text_train, text_test, img_train, img_test)
    print(json.dumps({"text_train": len(text_train), "text_test": len(text_test), "img_train": len(img_train),
                      "img_test": len(img_test)}), flush=True)

asyncio.run(main())
if "--train" in sys.argv:
    import pt_sft
    print(json.dumps(pt_sft.launch(), indent=1), flush=True)
