# Lab 21 — Links

**Họ tên:** Nguyễn Văn Việt

**MSSV:** 2A202602904

## Bài nộp

- GitHub repository: <https://github.com/little-duck-vie/Day21-Track3-NguyenVanViet-2A202602904-Finetuning-Lab>
- Google Colab RUN ALL: <https://colab.research.google.com/github/little-duck-vie/Day21-Track3-NguyenVanViet-2A202602904-Finetuning-Lab/blob/main/colab/Lab21_RUN_ALL.ipynb>

## Adapter công khai — Bonus B5

- Hugging Face: <https://huggingface.co/little-duck-vie/lab21-qwen35-triage-vi>
- Base model: `unsloth/Qwen3.5-4B`
- LoRA: `r=16`, `alpha=32`, placement `text-linear`

## Artefact bonus B1

- Kết quả merge: target `0.9700 → 0.9700` (`Δ = +0.0000`)
- Điều kiện: mức suy giảm sau merge không vượt quá `0.0100`
- Phục vụ nhiều adapter: hot-swap ít nhất hai adapter trên cùng một base

> Kết quả merge xác nhận thao tác đóng gói không làm giảm điểm target; verdict triển
> khai vẫn phải dựa trên cả target và regression gate trong `submission/REPORT.md`.
