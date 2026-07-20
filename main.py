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
    
    # Step 1: Search for videos
    search_request = youtube.search().list(
        part="snippet",
        q=query,
        maxResults=max_results,
        type="video"
    )
    search_response = search_request.execute()
    video_ids = [item["id"]["videoId"] for item in search_response["items"]]

    # Step 2: Get detailed stats
    stats_request = youtube.videos().list(
        part="snippet,statistics",
        id=",".join(video_ids)
    )
    stats_response = stats_request.execute()

    # Step 3: Structure the data
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
    
    # Combine title and description into one text field per video
    df["content"] = df["title"] + " " + df["description"]
    
    # Add user interest to the list of texts to compare
    all_text = [user_interest] + df["content"].tolist()
    
    # Apply TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_text)
    
    # Calculate cosine similarity between user interest and each video
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Add scores to dataframe and sort
    df["relevance_score"] = similarity_scores
    recommendations = df.sort_values("relevance_score", ascending=False).head(top_n)
    
    return recommendations

# --- Main program ---
print("🎬 YouTube AI Recommendation Tool")
print("----------------------------------")
user_interest = input("Enter your interests (e.g. 'machine learning for beginners'): ")

print("\nFetching videos from YouTube...")
df = get_youtube_data(user_interest)

print("Analyzing and ranking videos...\n")
recommendations = recommend_videos(user_interest, df)

print(f"🎯 Top 5 Recommendations for: '{user_interest}'")
print("=" * 60)
for i, row in recommendations.iterrows():
    print(f"\n#{list(recommendations.index).index(i)+1} {row['title']}")
    print(f"   Channel: {row['channel']}")
    print(f"   Views: {int(row['views']):,}")
    print(f"   Relevance Score: {row['relevance_score']:.2f}")
    print(f"   URL: {row['video_url']}")

# Save results
df.to_csv("youtube_data.csv", index=False)
print("\n✅ Data saved to youtube_data.csv")
