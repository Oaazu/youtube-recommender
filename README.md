# 🎬 YouTube AI Recommendation Tool

A Python-based web application that uses the YouTube Data API v3 to fetch live video data and an AI-powered recommendation engine to suggest relevant videos based on user interests.

🔗 **Live Demo:** [Click here to try it](https://youtube-recommender-i9bcmmdodvmw5tyyx6zqsn.streamlit.app)

---

## 📌 Project Overview

This tool allows users to enter their interests and receive personalized YouTube video recommendations ranked by relevance — not just popularity. Built as a portfolio project to demonstrate API integration, data handling, and applied AI/ML skills relevant to Business Analyst and AI/Business Systems Analyst roles.

---

## 🧠 How It Works

1. User enters their interests (e.g. "machine learning for beginners")
2. The app queries the YouTube Data API v3 and retrieves 20 videos
3. Video titles and descriptions are vectorized using TF-IDF
4. Cosine similarity scores are calculated between user input and each video
5. Top results are ranked by relevance and displayed with key metrics

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Data Collection | YouTube Data API v3 |
| Data Processing | pandas |
| AI/ML Engine | TF-IDF + Cosine Similarity (scikit-learn) |
| Web Application | Streamlit |
| Deployment | Streamlit Cloud |
| Version Control | Git & GitHub |

---

## 📊 Features

- 🔍 Live YouTube search based on user input
- 🤖 AI-powered relevance ranking (not just view count)
- 📈 Video metrics: views, likes, comments, relevance score
- ⚙️ Adjustable number of recommendations (3–10)
- 📋 Full dataset view with expandable table
- 🔗 Clickable video links directly to YouTube

---

## 🚀 Run Locally

1. Clone the repository
```bash
   git clone https://github.com/Oaazu/youtube-recommender.git
   cd youtube-recommender
```

2. Create and activate a virtual environment
```bash
   python -m venv venv
   venv\Scripts\activate
```

3. Install dependencies
```bash
   pip install -r requirements.txt
```

4. Create a `.env` file and add your YouTube API key
    YOUTUBE_API_KEY=your_api_key_here

5. Run the app
```bash
   streamlit run app.py
```

---

## 💡 Key Technical Decisions

- **TF-IDF over simple keyword matching** — captures word importance across the dataset, not just frequency
- **Relevance over popularity** — videos are ranked by content similarity to user input, not view count
- **Two-stage API call** — first fetches video IDs via search, then retrieves detailed statistics separately for richer data
- **API key security** — stored in `.env` locally and Streamlit Secrets in production, never exposed in code

---

## 👤 Author

**Jeffrey Arzu**  
[GitHub](https://github.com/Oaazu)