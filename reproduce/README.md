# Reproduce the paper

This folder reproduces the paper's tables with the public [`streamembed`](https://pypi.org/project/streamembed/) library and the [ScreenMemEval](https://huggingface.co/datasets/dayoon/ScreenMemEval) dataset. Run the commands from the top of the repository.

```
reproduce/
├── data.py              # downloads ScreenMemEval from Hugging Face
├── ocr.py               # Apple Vision OCR of every screenshot, for the text index (macOS)
├── accuracy.py          # Recall@10 of every row (runs encode.py, then evaluate.py)
├── latency.py           # seconds per screenshot of every row
├── encode.py            # encodes one stream of screenshots, the OCR texts, or the questions
├── evaluate.py          # scores them next to the paper's numbers
└── paper_numbers.json
```

```bash
uv pip install streamembed==0.3.1 huggingface_hub bm25s==0.3.11
python reproduce/data.py data            # ScreenMemEval, about 18 GB
python reproduce/ocr.py data             # on a Mac: OCR for the text index (uv pip install pyobjc-framework-Vision)
python reproduce/accuracy.py data        # Recall@10 of every row, on CUDA GPUs (--device mps on a Mac)
python reproduce/latency.py data         # seconds per screenshot, on an idle Mac (--cpu for the CPU rows)
```

Each command prints its table with the paper's number in parentheses and writes it to `results/`. `ocr.py` writes `data/<config>/ocr.jsonl` and continues where it stopped; copy these files to the GPU machine before `accuracy.py`, which leaves out the text index without them. `accuracy.py` runs one stream per GPU at a time and continues where it stopped when rerun; the vectors take about 4 GB.

| Row | Weights, preset |
|---|---|
| Text index (OCR + BM25, fine-tuned) | OCR texts embedded with the OCR LoRA, fused with BM25 |
| Screenshot (zero-shot) | base, `full` |
| Screenshot (LoRA fine-tuned) | LoRA, `full` |
| + Linear patch embed. | LoRA, `full+pe` (latency only) |
| + Spatial pruning | LoRA, `S+G` (with Grid Keep) |
| w/o Grid Keep | LoRA, `S` |
| + Temporal reuse | LoRA, `S+G+T+L` |
| + Consistency adapter = StreamEmbed | Consistency adapter, `S+G+T+L` |

Questions are embedded with the fine-tuned LoRA, except in the zero-shot row and the text index (the OCR LoRA). The text index is timed on the GPU only, as OCR plus the text embedding, since Apple Vision OCR runs on the Neural Engine. The paper encoded the screenshots in bf16 and streamembed uses fp16 on GPUs, so Recall@10 can differ slightly. The paper timed an AWS mac2-m2pro (Apple M2 Pro, 32 GB) and used `torch.compile` on the CPU, which streamembed does not, so CPU times will be slower.
