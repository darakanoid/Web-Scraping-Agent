# 🔍 Web Research Agent

A lightweight Streamlit app that performs real-time web research using **SerpAPI** for search and **Google Gemini Pro 1.5** for content analysis. It presents a concise report with confidence scores and verified sources — ideal for quick insights or academic overviews.

---

 📌 Features

- ✅ 5+ verified search results from SerpAPI  
- 🧠 Summarized answers generated using Gemini LLM  
- 🎯 Confidence scoring system (Low, Medium, High)  
- 📚 Source cards with titles, snippets, and external links  
- ✨ Clean, minimal Streamlit UI  



## 🛠️ Tech Stack

| Tool         | Purpose                         |
|--------------|---------------------------------|
| Python       | Backend logic                   |
| Streamlit    | Frontend UI                     |
| SerpAPI      | Web search API                  |
| Gemini Pro   | AI reasoning and content summarization |

---

 📂 Project Structure

---

![agent](https://github.com/user-attachments/assets/28d06759-1cb8-455b-812a-5297c73e1791)


--

---

## 📥 Installation

1. **Clone the repo**

bash
git clone https://github.com/yourusername/web-research-agent.git
cd web-research-agent
---
Create a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
---
pip install -r requirements.txt
---
streamlit run app.py
---
📌 Example Queries
"Top 5 recent breakthroughs in quantum computing"

"What are the latest updates on climate change policy?"

"Today's top headlines in technology"

🔒 API Usage
SerpAPI: Free plan gives 100 searches/month. Sign up

Google Gemini API: Requires a Google Cloud project and billing enabled. Docs

🧠 Future Improvements
🗂️ Export research reports to PDF

🔍 Search filter: News vs Web

🎨 Dark mode and UI themes

📊 Source trust score analysis

🤝 Contributing
PRs are welcome! If you find a bug or want to improve the tool, please feel free to send a pull request.

📄 License
MIT License – see LICENSE for details.

🌐 Credits
SerpAPI

Google Gemini

Streamlit

Built by Ankit Girish 
---


