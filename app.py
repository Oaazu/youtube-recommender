import streamlit as st
from googleapiclient.discovery import build
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd

# Load API key
load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")

# Build YouTube connection
youtube = build("youtube", "v3", developerKey=api_key)

def get_youtube_data(query, max_results=20):
    """Search YouTube and return structured video data."""
    search_request = youtube.search().list(
        part="snippet",
        q=query,
        maxResults=max_results,
        type="video"
    )
    search_response = search_request.execute()
    video_ids = [item["id"]["videoId"] for item in search_response["items"]]

    stats_request = youtube.videos().list(
        part="snippet,statistics",
        id=",".join(video_ids)
    )
    stats_response = stats_request.execute()

    videos = []
    for item in stats_response["items"]:
        video = {
            "title": item["snippet"]["title"].replace("&amp;", "&"),
            "channel": item["snippet"]["channelTitle"],
            "published_at": item["snippet"]["publishedAt"],
            "description": item["snippet"]["description"][:200],
            "views": item["statistics"].get("viewCount", 0),
            "likes": item["statistics"].get("likeCount", 0),
            "comments": item["statistics"].get("commentCount", 0),
            "video_url": f"https://www.youtube.com/watch?v={item['id']}"
        }
        videos.append(video)

    df = pd.DataFrame(videos)
    df["views"] = pd.to_numeric(df["views"])
    df["likes"] = pd.to_numeric(df["likes"])
    df["comments"] = pd.to_numeric(df["comments"])
    df["published_at"] = pd.to_datetime(df["published_at"])
    return df

def recommend_videos(user_interest, df, top_n=5):
    """Recommend videos based on user interest using TF-IDF."""
    df["content"] = df["title"] + " " + df["description"]
    all_text = [user_interest] + df["content"].tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_text)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    df["relevance_score"] = similarity_scores
    recommendations = df.sort_values("relevance_score", ascending=False).head(top_n)
    return recommendations

# --- Streamlit UI ---
st.set_page_config(page_title="YouTube AI Recommender", page_icon="🎬", layout="wide")

st.title("🎬 YouTube AI Recommendation Tool")
st.markdown("Enter your interests below and get personalized YouTube video recommendations powered by AI.")

# Sidebar
st.sidebar.header("⚙️ Settings")
num_results = st.sidebar.slider("Number of recommendations", min_value=3, max_value=10, value=5)
st.sidebar.markdown("---")
st.sidebar.markdown("Built with YouTube Data API v3 + TF-IDF AI Engine")

# Main input
user_interest = st.text_input("🔍 What do you want to learn or watch today?", placeholder="e.g. machine learning for beginners")

if st.button("Get Recommendations"):
    if user_interest:
        with st.spinner("Fetching videos from YouTube..."):
            df = get_youtube_data(user_interest)
        
        with st.spinner("Analyzing and ranking videos with AI..."):
            recommendations = recommend_videos(user_interest, df, top_n=num_results)

        st.success(f"Top {num_results} recommendations for: **{user_interest}**")
        st.markdown("---")

        for i, (_, row) in enumerate(recommendations.iterrows()):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### #{i+1} [{row['title']}]({row['video_url']})")
                st.markdown(f"📺 **Channel:** {row['channel']}")
                st.markdown(f"📝 {row['description']}")
            with col2:
                st.metric("Views", f"{int(row['views']):,}")
                st.metric("Likes", f"{int(row['likes']):,}")
                st.metric("Relevance", f"{row['relevance_score']:.2f}")
            st.markdown("---")

        # Show full data table
        with st.expander("📊 View full dataset"):
            st.dataframe(df[["title", "channel", "views", "likes", "comments", "relevance_score"]].sort_values("relevance_score", ascending=False))

    else:
        st.warning("Please enter your interests first!")