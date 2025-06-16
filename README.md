Dưới đây là bản tóm tắt quy trình và các kỹ thuật được đề xuất trong báo cáo để thực hiện dự án phát hiện clickbait.

### **Mục tiêu Dự án**

[cite_start]Mục tiêu chính là khai thác Mô hình Ngôn ngữ Lớn (LLM) để phát hiện clickbait[cite: 4]. [cite_start]Dự án sẽ thực hiện phân tích so sánh chi tiết giữa hai phương pháp chính[cite: 5]:
* [cite_start]**Tinh chỉnh (Fine-tuning):** Điều chỉnh một mô hình ngôn ngữ đã được tiền huấn luyện có kích thước vừa phải trên một tập dữ liệu chuyên biệt[cite: 5].
* [cite_start]**Kỹ thuật gợi ý (Prompting):** Sử dụng các LLM nền tảng cực lớn thông qua các câu lệnh gợi ý (prompt) mà không cần huấn luyện lại[cite: 7, 8].

[cite_start]Toàn bộ các thí nghiệm sẽ được thực hiện trên tập dữ liệu tiêu chuẩn **Webis-Clickbait-17**[cite: 9]. [cite_start]Mục tiêu cuối cùng là đưa ra các khuyến nghị chiến lược dựa trên hiệu quả, chi phí, tốc độ và tính khả thi của mỗi phương pháp[cite: 10].

### **Tập dữ liệu và Tiền xử lý (Webis-Clickbait-17)**

[cite_start]Quy trình xử lý dữ liệu từ tập Webis-Clickbait-17 là bước nền tảng cho cả hai hướng tiếp cận[cite: 42].

1.  **Phân chia Tập dữ liệu:**
    * [cite_start]Do tập kiểm thử chính thức chỉ có thể truy cập qua nền tảng TIRA [cite: 48, 77][cite_start], cần phải tự tạo một tập kiểm định (validation set)[cite: 77].
    * [cite_start]Chia tập huấn luyện có sẵn (`clickbait17-train-170630.zip`) thành hai phần: một phần để huấn luyện (ví dụ: 80-90%) và một phần để kiểm định (10-20%)[cite: 78].
    * [cite_start]Thực hiện phân chia có tính phân tầng (stratified split) để đảm bảo tỷ lệ các lớp trong hai tập là tương đương nhau[cite: 79].

2.  **Tạo Nhãn Phân loại:**
    * [cite_start]Chuyển đổi điểm số liên tục `truthMean` của dữ liệu thành nhãn nhị phân (`clickbait` và `no-clickbait`)[cite: 54, 55].
    * [cite_start]Một ngưỡng phổ biến để phân chia là `0.5`: các mẫu có `truthMean >= 0.5` được gán nhãn là `clickbait`[cite: 56]. [cite_start]Dự án này sẽ sử dụng phương pháp phân loại nhị phân[cite: 62].

3.  **Làm sạch Văn bản:**
    * [cite_start]Áp dụng các bước làm sạch tiêu chuẩn cho văn bản từ trường `postText`[cite: 63].
    * [cite_start]Các bước bao gồm: chuyển thành chữ thường [cite: 64][cite_start], loại bỏ URL, đề cập người dùng (@username) và các ký tự đặc biệt [cite: 65][cite_start], và xử lý các hashtag[cite: 66].

4.  **Tokenization:**
    * [cite_start]**Đối với Fine-tuning:** Bắt buộc sử dụng `tokenizer` tương ứng với mô hình đã chọn (ví dụ: `RobertaTokenizerFast` cho RoBERTa)[cite: 68]. [cite_start]Quá trình này chuyển văn bản thành `input_ids` và `attention_mask`, đồng thời thực hiện cắt ngắn (truncate) hoặc đệm (pad) các chuỗi để có cùng độ dài[cite: 69].
    * [cite_start]**Đối với Prompting:** Không cần tokenization thủ công; văn bản thô được đưa trực tiếp vào prompt và LLM sẽ tự xử lý[cite: 70].

5.  **Xử lý Mất cân bằng Lớp:**
    * [cite_start]Tập dữ liệu bị mất cân bằng với tỷ lệ clickbait/không-clickbait khoảng 1:3[cite: 47].
    * [cite_start]Áp dụng các kỹ thuật như lấy mẫu lại (oversampling lớp thiểu số, undersampling lớp đa số) [cite: 72, 73] [cite_start]hoặc gán trọng số lớp (class weights) cao hơn cho lớp thiểu số trong hàm mất mát để "phạt" nặng hơn các lỗi dự đoán sai trên lớp này[cite: 74, 75].

### **Hướng tiếp cận 1: Tinh chỉnh (Fine-tuning)**

[cite_start]Đây là phương pháp chuyên môn hóa một mô hình cho tác vụ phát hiện clickbait[cite: 83].

1.  **Lựa chọn Mô hình:**
    * [cite_start]**RoBERTa (`roberta-base`):** Lựa chọn hàng đầu vì đã có bằng chứng cho thấy nó đạt hiệu suất SOTA trên tập dữ liệu Webis-Clickbait-17[cite: 88, 94].
    * [cite_start]**Mistral-7B / LLaMA-3-8B:** Các mô hình thế hệ mới, hiệu quả cao, là ứng viên xuất sắc để khám phá sự cân bằng giữa hiệu suất và chi phí tính toán[cite: 89, 90].
    * [cite_start]**Chiến lược đề xuất:** Bắt đầu với `RoBERTa-base` và sau đó thử nghiệm với `Mistral-7B`[cite: 95].

2.  **Quy trình Fine-tuning:**
    * [cite_start]Sử dụng thư viện `transformers` của HuggingFace, đặc biệt là API `Trainer`[cite: 96].
    * [cite_start]Các bước bao gồm: tải dữ liệu đã tiền xử lý, tải `AutoTokenizer` và `AutoModelForSequenceClassification` phù hợp, định nghĩa hàm `compute_metrics` để tính toán các chỉ số (Accuracy, F1-score), cấu hình các đối số huấn luyện (`TrainingArguments`), và khởi tạo `Trainer` để bắt đầu quá trình huấn luyện[cite: 98, 99, 100, 101].

3.  **Kỹ thuật Tối ưu Hóa:**
    * [cite_start]**PEFT (Parameter-Efficient Fine-Tuning) / LoRA (Low-Rank Adaptation):** Thay vì tinh chỉnh toàn bộ mô hình, chỉ tinh chỉnh một phần nhỏ các tham số (<1%) bằng cách chèn các ma trận có hạng thấp (low-rank) có thể huấn luyện[cite: 103, 104, 105]. [cite_start]Kỹ thuật này giúp giảm đáng kể bộ nhớ GPU cần thiết[cite: 106].
    * [cite_start]**Unsloth:** Một thư viện tối ưu hóa giúp tăng tốc độ huấn luyện cho các mô hình LLaMA và Mistral lên đến 2 lần và giảm 60% mức sử dụng bộ nhớ[cite: 108, 109].
    * [cite_start]**Kết hợp đề xuất:** Sử dụng `Mistral-7B` với `PEFT/LoRA` và `Unsloth` được xem là phương pháp fine-tuning hiện đại và hiệu quả nhất[cite: 111].

### **Hướng tiếp cận 2: Kỹ thuật Gợi ý (Prompting)**

[cite_start]Phương pháp này điều khiển các LLM nền tảng mà không cần huấn luyện lại[cite: 117].

1.  **Thiết kế Prompt:**
    * [cite_start]**Zero-shot Prompting:** Cung cấp cho mô hình mô tả tác vụ mà không có ví dụ nào[cite: 121]. [cite_start]Prompt cần có chỉ dẫn rõ ràng, dữ liệu đầu vào, và chỉ báo đầu ra, đồng thời yêu cầu mô hình trả lời theo một định dạng cố định (ví dụ: chỉ "Clickbait" hoặc "Not Clickbait")[cite: 122, 123, 124, 127].
    * [cite_start]**Few-shot Prompting:** Cung cấp cho mô hình một vài ví dụ (cặp đầu vào-đầu ra) ngay trong prompt[cite: 129]. [cite_start]Các ví dụ nên đa dạng, bao gồm cả các trường hợp rõ ràng và các trường hợp ranh giới (borderline) để giúp mô hình học được quy luật[cite: 131, 132].

2.  **Lựa chọn Mô hình:**
    * [cite_start]Hiệu quả phụ thuộc vào năng lực của LLM[cite: 137].
    * [cite_start]Các ứng viên hàng đầu bao gồm: `GPT-4`/`GPT-4o` (OpenAI) [cite: 139][cite_start], `Claude 3` (Anthropic) [cite: 140][cite_start], và các mô hình mã nguồn mở hiệu năng cao như `Mixtral` hoặc `LLaMA 3`[cite: 141].

3.  **Quy trình Đánh giá:**
    * [cite_start]Sử dụng tập kiểm định đã tạo ở bước tiền xử lý[cite: 143].
    * [cite_start]Viết kịch bản để lặp qua từng mẫu trong tập kiểm định, xây dựng prompt tương ứng, và gửi yêu cầu đến API của LLM[cite: 144].
    * [cite_start]Thu thập các dự đoán và so sánh chúng với nhãn thật để tính toán các thước đo hiệu suất[cite: 147].

### **Đánh giá và So sánh**

1.  **Thước đo Hiệu suất:**
    * [cite_start]Sử dụng các thước đo phân loại tiêu chuẩn: `Accuracy`, `Precision`, `Recall`, và `F1-score`[cite: 148].
    * [cite_start]Sử dụng ma trận nhầm lẫn (Confusion Matrix) để phân tích các loại lỗi mà mỗi phương pháp mắc phải[cite: 149].

2.  **Mục tiêu Benchmark:**
    * [cite_start]Kết quả từ Clickbait Challenge 2017 cho thấy mô hình chiến thắng (goldfish) đạt `Accuracy` 0.876 và `F1-score` 0.741[cite: 188, 190].
    * [cite_start]Các nghiên cứu sau đó sử dụng Transformer đã đạt được kết quả cao hơn, ví dụ một nghiên cứu với RoBERTa báo cáo `Accuracy` 0.8575 và `F1-score` 0.6901 trên tập kiểm định riêng của họ[cite: 193].
    * [cite_start]Mục tiêu của dự án là mô hình fine-tuning phải đạt hiệu suất tương đương hoặc vượt qua các kết quả này[cite: 197].

### **Chiến lược Triển khai Đề xuất**

[cite_start]Báo cáo đề xuất một chiến lược kết hợp theo hai giai đoạn cho việc phát triển một sản phẩm thực tế[cite: 173, 212]:

1.  **Giai đoạn 1: Tạo mẫu thử (Prototyping)**
    * [cite_start]Sử dụng phương pháp **prompting** với một API LLM mạnh như `GPT-4o`[cite: 174, 214].
    * [cite_start]Mục đích là để nhanh chóng xây dựng một phiên bản chứng minh khái niệm (Proof of Concept) và xác thực ý tưởng với chi phí và thời gian tối thiểu[cite: 175, 177, 215].

2.  **Giai đoạn 2: Phát triển Sản phẩm (Production)**
    * [cite_start]Khi sản phẩm cần triển khai ở quy mô lớn, chuyển sang phương pháp **fine-tuning**[cite: 177, 218].
    * **Stack công nghệ đề xuất:**
        * [cite_start]**Mô hình:** `Mistral-7B`[cite: 178, 219].
        * [cite_start]**Kỹ thuật:** `PEFT/LoRA`[cite: 179, 220].
        * [cite_start]**Thư viện:** `Unsloth`[cite: 179, 221].
    * [cite_start]Mục đích là để tạo ra một mô hình được tối ưu hóa, hiệu quả về chi phí, có tốc độ suy luận nhanh và đảm bảo quyền riêng tư dữ liệu khi vận hành[cite: 172, 181, 222].