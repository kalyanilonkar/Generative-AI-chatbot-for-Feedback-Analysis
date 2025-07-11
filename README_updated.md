
# 🎓 Generative AI Chatbot for Student Feedback Analysis

This project builds a **Generative AI-powered chatbot** that helps instructors analyze qualitative student feedback, understand confusion areas, and improve teaching effectiveness. It leverages both **traditional machine learning models** and **Large Language Models (LLMs)** (Mistral, Falcon) alongside **Power BI dashboards** for a comprehensive data-driven educational insight system.

---

##  Features

**Sentiment Classification:** Classifies student comments into 1–5 star ratings using:

**Traditional ML models (Random Forest, Logistic Regression)
Pretrained BERT sentiment model (nlptown/bert-base-multilingual-uncased)**

**Generative Q&A:** Uses local Mistral LLM to answer questions like:

“Why were students confused?”
“What improvements can be made?”
“Which topics were students most confident about?”
Power BI Dashboards: Automatically saves processed data to CSV, enabling dashboards that visualize:
Sentiment trends
Session-wise performance
Common areas of confusion

**Streamlit Chatbot** UI: Lets instructors chat with the system and receive data-backed responses.

**Project Goal**
To help instructors derive actionable insights from feedback, by experimenting with and comparing:
Traditional ML sentiment models (which excel at quick classification & quant metrics)
LLM-based reasoning (which excel at generating explanations & qualitative insights)


##  Tech Stack

- **Python** (pandas, sklearn, matplotlib, seaborn, nltk, regex)
- **Hugging Face Transformers** for LLM-based sentiment and QA (Mistral, Falcon)
- **Streamlit** for interactive chatbot UI
- **Power BI** for dynamic sentiment and confusion dashboards
- **Google Colab** for GPU-based LLM experiments and fine-tuning

---

##  Project Structure

```
data/               # Sample student feedback CSVs
notebooks/          # Jupyter notebooks for EDA, ML, and LLM experiments
app/
  ├── chatbot_streamlit.py  # Streamlit chatbot
  ├── requirements.txt
  └── assets/               # Images for UI & README
dashboards/
  └── powerbi_dashboard.pbix # Power BI file
```

---

##  Demo Screenshots

<img src="app/assets/dashboard.png" width="450">

---

##  Example: Using Mistral LLM in Google Colab

We used the free Google Colab GPU environment to run open-source local LLMs like **Mistral-7B** for generative feedback analysis.

```python
!pip install -q transformers accelerate bitsandbytes

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16
)

prompt = "Why were students confused in the lecture?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

--->> This allows asking open-ended questions on student feedback — replacing traditional models with **generative reasoning** to identify confusion areas, engagement issues, or improvement points.

---

##  Getting Started

1] Clone this repository
```bash
git clone https://github.com/yourusername/genai-student-feedback-chatbot.git
cd genai-student-feedback-chatbot
```

2] Install dependencies
```bash
pip install -r app/requirements.txt
```

3] Run Streamlit app
```bash
cd app
streamlit run chatbot_streamlit.py
```

4] Open Power BI and connect to `powerbi_output.csv` for live dashboards.

---

##  Future Enhancements

- Fine-tune LLM on domain-specific instructor feedback  
- Incorporate multilingual support for global classrooms  
- Add continuous improvement loop using instructor feedback (RLHF concepts)

---

##  License

[MIT License](LICENSE)

---

##  Author

**Kalyani Lonkar**  
🎓 MSc Business Analytics | University of Manchester  
📫 [LinkedIn](https://linkedin.com/in/your-link) • ✉️ [Email](mailto:your-email@domain.com)

---

##  Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Streamlit](https://streamlit.io)
- [Microsoft Power BI](https://powerbi.microsoft.com)
