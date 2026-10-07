# Lab 21 — Báo cáo đánh giá Fine-tuning

**Họ tên:** Nguyễn Văn Việt

**MSSV:** 2A202602904

**Ngày thực hiện:** 07/10/2026

**Tier:** T4 · **Base model:** `unsloth/Qwen3.5-4B` · **GPU:** NVIDIA T4 16 GB

> Báo cáo này được tổng hợp trực tiếp từ các artefact trong `results/`. Lần chạy hiện
> tại bật `EVAL_LIMIT=8` (`smoke_mode=true`) cho baseline và verdict tổng hợp, vì vậy
> các kết quả đó chỉ có ý nghĩa như một phép thử nhanh. File `qualitative.json` được tạo
> sau đó có đủ 50 mẫu target, nhưng không chứa prediction của baseline (b); hai nguồn
> artefact này chưa tạo thành một phép so sánh full-eval hoàn chỉnh để quyết định deploy.

## 1. Bài toán, dữ liệu và cấu hình

Bài toán là phân loại ticket chăm sóc khách hàng tiếng Việt thành một JSON có đúng bốn
trường: `intent`, `urgency`, `product` và `sentiment`. Tôi dùng bộ dữ liệu mặc định gồm
250 ticket vì dữ liệu này bám sát đầu ra có cấu trúc cần học, đủ lớn cho một thí nghiệm
LoRA ngắn, đồng thời có nhãn theo từng trường để đo chính xác hơn exact match. Dữ liệu
được chia cố định bằng seed 42 thành 225 mẫu train và 25 mẫu validation (90/10). Tập
đánh giá độc lập có 50 mẫu target và 15 mẫu regression, nhưng lần chạy hiện tại chỉ lấy
8 mẫu đầu của mỗi tập.

Tôi chọn `unsloth/Qwen3.5-4B` vì đây là model mặc định phù hợp tier T4: đủ năng lực xử
lý tiếng Việt và sinh JSON, trong khi LoRA 16-bit vẫn nằm trong giới hạn bộ nhớ. Cấu
hình chính dùng LoRA `r=16`, `alpha=32`, gắn vào 12 lớp tuyến tính văn bản, LR `1e-4`,
`MASK_MODE=assistant-only`, batch hiệu dụng 16, 2 epoch tương ứng 30 optimizer step.

| Thuộc tính | Giá trị |
|---|---:|
| Số mẫu dữ liệu | 250 |
| Train / validation | 225 / 25 (seed 42) |
| Độ dài token mean / p95 / max | 93.1 / 98 / 101 |
| `suggested_max_length` | 256 |
| `max_length` thực dùng theo tier T4 | 1024 |
| Mask | `assistant-only` |
| Epoch / optimizer step | 2 / 30 |

`max_length=1024` lớn hơn mức 256 được gợi ý từ p95. Đây là giá trị cố định của cấu
hình tier T4, được giữ giống nhau ở cả bốn run để phép đối chứng không thay đổi thêm
một biến. Tuy vậy, với corpus hiện tại, 256 hợp lý và tiết kiệm hơn vì ngay cả mẫu dài
nhất chỉ có 101 token; lần chạy chính thức nên đổi đồng bộ sang 256 và đo lại.

Chat template **có giữ nguyên khối `<think>`**: `template_check.json` ghi nhận đủ thẻ
mở, nội dung reasoning và kết luận `reasoning preserved — safe to train on traces`.

## 2. Bằng chứng loss mask

Kết quả mask xác nhận chỉ phần trả lời của assistant tham gia vào loss:

| Chỉ số | Kết quả |
|---|---:|
| Token được giám sát / tổng token | 39 / 94 |
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi bị mask khỏi loss | `true` |

Đoạn đầu của phần được tính loss:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Tỷ lệ 41.49% cũng loại trừ lỗi phổ biến là tính loss trên gần như toàn bộ prompt. Nhờ
đó, model học cấu trúc và nhãn của câu trả lời thay vì học chép lại system/user message.

## 3. Mốc đánh giá đóng băng và kết quả chính

Baseline được đo và đóng băng trước khi train. Prompt tối ưu là prompt nguyên bản của
lab, nhận diện bởi SHA `719e74d3b6232053`; tôi không chỉnh sửa prompt này. Trên lát cắt
8 mẫu, prompt tối ưu thực sự mạnh hơn prompt ngây thơ: target tăng 0.6875 và format
tăng từ 0 lên 1.0, trong khi regression giữ nguyên 0.75.

| Run | Target | Regression | Format | Latency (ms) | n |
|---|---:|---:|---:|---:|---:|
| (a) Base + naive prompt | 0.0000 | 0.7500 | 0.0000 | 3577.7 | 8 |
| (b) Base + optimized prompt | 0.6875 | 0.7500 | 1.0000 | 1043.1 | 8 |
| (c) LoRA fine-tune | **0.9375** | 0.6250 | 1.0000 | 1599.4 | 8 |

Fine-tune tăng target thêm 0.25 so với baseline (b), giữ format hoàn hảo, nhưng giảm
regression 0.125 và chậm hơn khoảng 556.3 ms mỗi mẫu. Điều này cho thấy adapter đã học
tốt tác vụ hẹp, song đổi lại bằng năng lực tổng quát và độ trễ.

## 4. Giải phẫu ba cấu hình đối chứng

Cả bốn run đều dùng đúng 30 step. Mỗi đối chứng chỉ thay đổi một trục so với `correct`:
`attn_only` đổi vị trí gắn adapter và tăng rank để khớp ngân sách tham số; `wrong_lr`
chỉ giảm LR từ `1e-4` xuống `1e-5`; `qlora` chỉ chuyển base từ 16-bit sang 4-bit.

| Run | Vị trí | r | Trainable params | LR | Final loss | Target | Format | VRAM (GB) | Train (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6259 | **0.9375** | 1.0 | 8.78 | 431.4 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | **0.5364** | **0.9375** | 1.0 | 8.79 | 291.5 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | 0.0000 | 0.0 | 8.78 | 431.4 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.8438 | 1.0 | **3.86** | 515.6 |

### 4.1. Vị trí adapter so với rank

`attn_only` và `correct` gần như có cùng ngân sách: 32,456,704 so với 32,464,896 tham
số, chỉ lệch 8,192 tham số (khoảng 0.025%). Trên target, hai cấu hình hòa ở 0.9375,
trong khi `attn_only` có final loss thấp hơn, 0.5364 so với 0.6259. Vì vậy, xếp hạng
theo train loss sẽ nói `attn_only` thắng, còn thước đo tác vụ chỉ cho phép kết luận hòa.
Thí nghiệm này không chứng minh rank lớn hơn tạo chất lượng target tốt hơn; với tác vụ
triage hẹp và lát cắt nhỏ, gắn q,v ở rank đã khớp ngân sách đạt cùng kết quả với việc
phủ các lớp text-linear. Vị trí gắn adapter vẫn là biến cần đo trực tiếp, còn rank không
thể được xem là đòn bẩy độc lập khi ngân sách tham số chưa được kiểm soát.

### 4.2. Learning rate

`wrong_lr` chỉ khác một con số: LR giảm 10 lần xuống thang full fine-tuning `1e-5`.
Final loss của nó dừng ở 1.5702, cao hơn 0.6259 của run đúng, và target/format cùng rơi
về 0. Nếu chỉ nhìn loss mà không biết LR, tôi có thể kết luận nhầm rằng model, mask hoặc
dữ liệu không học được. Đối chứng này chỉ ra rằng LoRA cần LR đúng thang; tăng rank hay
đổi vị trí không giải quyết được một bước cập nhật quá nhỏ trong ngân sách chỉ 30 step.

### 4.3. QLoRA

QLoRA giảm peak VRAM từ 8.78 GB xuống 3.86 GB, tiết kiệm 4.92 GB, tương đương khoảng
56.0%. Đổi lại, target giảm 0.0937, final loss tăng 0.0799, thời gian train tăng 84.2
giây và latency suy luận tăng 422.8 ms so với `correct`. Trên số đo hiện có, đánh đổi
này ủng hộ khuyến nghị không dùng QLoRA cho Qwen3.5 khi T4 vẫn chứa được LoRA 16-bit.
Nếu phần cứng chỉ có khoảng 4–6 GB VRAM thì QLoRA vẫn là phương án khả dụng, nhưng đó
là lựa chọn vì giới hạn bộ nhớ chứ không phải lựa chọn cho chất lượng hoặc tốc độ.

## 5. Phán quyết của cổng hồi quy

**Kết quả: FAILED**

`target Δ = +0.250` · `regression Δ = -0.125` · `valid_trace_rate = 0.0000`

Fine-tune vượt mốc prompt tối ưu rõ rệt trên target (0.9375 so với 0.6875), nên adapter
đã học được hành vi triage chuyên biệt. Tuy nhiên, điều kiện triển khai không chỉ là cải
thiện target: regression chỉ được phép giảm tối đa 0.020, trong khi số đo thực tế giảm
0.125. Mức tụt này lớn hơn ngưỡng cho phép hơn sáu lần, vì vậy verdict `FAILED` là hợp
lý và không nên được “sửa” bằng cách nới gate hay làm yếu baseline. Format vẫn đạt 1.0,
nên lỗi không nằm ở JSON/template; bằng chứng mask cũng đúng, nên nguyên nhân phù hợp
nhất là quên thảm họa khi 225 mẫu huấn luyện đều tập trung vào một tác vụ hẹp. Hướng
khắc phục là trộn khoảng 1–5% replay data tổng quát, huấn luyện lại rồi đo trên toàn bộ
tập eval. `valid_trace_rate=0` còn cho thấy model không sinh reasoning trace hợp lệ,
nhưng tác vụ yêu cầu chỉ trả JSON nên chỉ số này không phải nguyên nhân trực tiếp khiến
cổng thất bại. Quan trọng hơn, toàn bộ verdict hiện chỉ dựa trên 8 mẫu mỗi nhóm; nó là
tín hiệu chẩn đoán, chưa phải ước lượng đủ chắc chắn.

## 6. Phân tích định tính

`results/qualitative.json` chứa đủ 50 mẫu target, được sắp từ điểm thấp đến cao. Có 44
mẫu đạt 1.00 và 6 mẫu đạt 0.75; không có mẫu dưới 0.75. Trung bình field accuracy suy
ra trực tiếp từ file là `(44 × 1 + 6 × 0.75) / 50 = 0.9700`. Bảng dưới chọn ba ca đúng
hoàn toàn và hai trong sáu ca sai, thay vì chỉ trình bày các trường hợp thuận lợi.

| # | Ticket (rút gọn) | Nhãn đúng | Prompt (b) | Fine-tune (c) | Nhận xét |
|---|---|---|---|---|---|
| 1 (`i=0`) | Trả chuột không dây, “Gấp”, shop hỗ trợ tốt | `doi_tra`, `cao`, chuột không dây, `tich_cuc` | Không được lưu trong artefact | `doi_tra`, `cao`, chuột không dây, `tich_cuc` | ✅ FT đúng 4/4 trường |
| 2 (`i=2`) | Hoàn tiền đèn bàn LED, “Quá hạn rồi”, cảm ơn shop | `hoan_tien`, `cao`, đèn bàn LED, `tich_cuc` | Không được lưu trong artefact | `hoan_tien`, `cao`, đèn bàn LED, `tich_cuc` | ✅ FT đúng 4/4 trường |
| 3 (`i=8`) | Hỏi bảo hành chuột không dây, “Không vội”, vẫn tin tưởng | `hoi_thong_tin`, `thap`, chuột không dây, `tich_cuc` | Không được lưu trong artefact | `hoi_thong_tin`, `thap`, chuột không dây, `tich_cuc` | ✅ FT đúng 4/4 trường |
| 4 (`i=3`) | Chưa thấy tiền hoàn bình giữ nhiệt, “Khi nào tiện” | `hoan_tien`, `thap`, bình giữ nhiệt, `tich_cuc` | Không được lưu trong artefact | `hoan_tien`, **`trung_binh`**, bình giữ nhiệt, `tich_cuc` | ❌ FT sai `urgency` (3/4) |
| 5 (`i=5`) | Nồi chiên thiếu phụ kiện, “Khi nào tiện” | `san_pham_loi`, `thap`, nồi chiên không dầu, `trung_tinh` | Không được lưu trong artefact | `san_pham_loi`, **`trung_binh`**, nồi chiên không dầu, `trung_tinh` | ❌ FT sai `urgency` (3/4) |

Hai ca thua được trình bày đều nhận diện đúng intent và product nhưng nâng urgency từ
`thap` lên `trung_binh`. Cả hai cùng chứa cụm “Khi nào tiện”, cho thấy model chưa học
ổn định rằng đây là tín hiệu yêu cầu không gấp. Bốn trong sáu ca 0.75 còn lại cũng được
xếp ở đầu `qualitative.json`; ba ca `i=12`, `i=39` và `i=46` thể hiện cùng dự đoán
`urgency=trung_binh` ở phần output được lưu, củng cố giả thuyết rằng urgency là điểm yếu
chính. Đây là lỗi có cấu trúc chứ không phải lỗi JSON: format tổng hợp vẫn bằng 1.0.

Một giới hạn quan trọng là notebook chỉ lưu `ft_pred` đã cắt ở 90 ký tự và không lưu
prediction theo từng mẫu của baseline (b). Vì thế bảng dùng `ft_score` cùng nhãn gốc để
xác định thắng/thua so với ground truth, nhưng chưa thể khẳng định ở từng ticket rằng
fine-tune tốt hơn hay kém hơn prompt tối ưu. Muốn so sánh cặp đúng nghĩa, lần chạy NB5
tiếp theo cần lưu thêm `baseline_b_pred` và output đầy đủ, không cắt chuỗi.

## 7. Kết luận và điều tôi học được

Tôi **chưa nên deploy** adapter này. Về mặt tích cực, cấu hình LoRA đúng đã chuyển hành
vi triage vào trọng số: chỉ với prompt ngắn, target đạt 0.9375 và đầu ra đúng schema ở
100% mẫu thử. Đối chứng LR cho bằng chứng nhân quả mạnh nhất: khi chỉ hạ LR mười lần,
target và format cùng về 0, nên thang learning rate là điều kiện cần trong ngân sách
30 step. Đối chứng vị trí lại cho một kết luận tinh tế hơn: dù `attn_only` dùng rank 283
và có train loss tốt hơn, nó chỉ hòa `correct` trên target. Vì vậy không thể dùng train
loss hoặc rank lớn để thay cho đánh giá end-to-end; vị trí adapter phải được so trong
cùng ngân sách tham số. Mask đúng là nền móng giúp thí nghiệm có nghĩa, còn chất lượng
và độ phủ dữ liệu quyết định khả năng giữ năng lực ngoài miền. Chính điểm này làm bản
fine-tune thất bại ở regression: target tăng 0.25 nhưng regression mất 0.125. Phân tích
định tính đủ 50 mẫu cho target trung bình 0.9700 và chỉ ra lỗi urgency có cấu trúc, nhưng
baseline/verdict tổng hợp vẫn là smoke run 8 mẫu nên độ bất định của phép so sánh còn lớn.
Trước khi cân nhắc triển khai, tôi sẽ thêm 1–5% replay data tổng quát, dùng `max_length`
256, huấn luyện lại, chạy đủ 50 mẫu target và 15 mẫu regression, rồi kiểm tra thủ công
năm trường hợp gồm cả ca thắng lẫn ca thua. Chỉ khi vượt cổng hồi quy trên tập đầy đủ
và không xuất hiện lỗi định tính nghiêm trọng thì lợi ích target mới đủ thuyết phục.

### Ba điều tôi học được

1. Train loss thấp không đồng nghĩa năng lực tác vụ tốt hơn: `attn_only` có loss 0.5364
   nhưng chỉ hòa `correct` ở target 0.9375.
2. Với LoRA ngắn, LR là đòn bẩy quyết định: đổi riêng `1e-4` thành `1e-5` làm target từ
   0.9375 xuống 0 và phá luôn định dạng JSON.
3. Fine-tune có thể thắng lớn trong miền nhưng vẫn không đạt điều kiện triển khai:
   target tăng 0.25 đồng thời regression giảm 0.125, nên luôn cần regression gate.

**Nếu có thêm 2 giờ**, tôi sẽ thêm replay data tổng quát ở các mức 1%, 3% và 5%, giữ
nguyên các siêu tham số còn lại, chọn mức nhỏ nhất vượt regression gate, sau đó chạy
toàn bộ eval và xuất `qualitative.json` để phân tích lỗi theo intent, urgency và product.

## Phụ lục — Hạng mục thưởng

### B1 — Merge và hot-swap nhiều adapter (+3)

Tôi chạy NB6 trên toàn bộ tập target sau khi đã có các adapter `correct`, `attn_only`
và `qlora`. Kết quả kiểm tra merge:

| Chỉ số | Giá trị |
|---|---:|
| Target trước merge | 0.9700 |
| Target sau merge | 0.9700 |
| Delta | +0.0000 |
| Mức suy giảm tối đa cho phép | 0.0100 |

Điểm target được giữ nguyên hoàn toàn sau `merge_and_unload()`, nên phép merge vượt
điều kiện không được tụt quá 0.01. Tôi cũng đã nạp nhiều adapter trên cùng một base và
chuyển adapter theo request bằng `set_adapter()`. Điều này xác nhận hai phương án phục
vụ có đánh đổi khác nhau: model merged không còn overhead LoRA lúc suy luận nhưng mỗi
tác vụ cần một bản model đầy đủ và không thể hot-swap; giữ adapter riêng tốn thêm một
ít chi phí suy luận nhưng chỉ cần một base trong VRAM, dễ chuyển tác vụ, rollback và
phục vụ nhiều khách hàng. Kết quả merge tốt không thay đổi verdict triển khai ở phần 5:
nó chứng minh tính đúng đắn của thao tác đóng gói, không khắc phục suy giảm regression.

### B5 — Hugging Face Hub (+2)

Adapter LoRA `correct` đã được tải lên Hugging Face Hub ở chế độ công khai:

<https://huggingface.co/little-duck-vie/lab21-qwen35-triage-vi>

Repository công khai chứa cấu hình PEFT và trọng số adapter để người chấm có thể tải
lại cùng base model `unsloth/Qwen3.5-4B`. Việc chỉ phát hành adapter giúp artefact nhỏ
hơn đáng kể so với model đã merge, đồng thời giữ rõ quan hệ giữa base và phần trọng số
được fine-tune.

- [x] B1 — merge, kiểm tra điểm không tụt và hot-swap ít nhất hai adapter
- [ ] B2 — dataset miền riêng có mô tả khử nhiễm
- [ ] B3 — so sánh hai chế độ mask cho reasoning-trace collapse
- [ ] B4 — quét rank có kiểm soát
- [x] B5 — công khai adapter trên Hugging Face Hub
