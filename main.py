from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from prediction import PredictionPipeline

app = FastAPI(
    title="Semantic Book Recommender API",
    description="Semantic book recommendations using FAISS + Cross-Encoder",
    version="1.0"
)

pipeline = PredictionPipeline()


class BookRecommendation(BaseModel):
    title: str
    authors: str
    description: str
    thumbnail: Optional[str]
    category: Optional[str]



@app.get("/recommend", response_model=List[BookRecommendation])
def recommend_books(
    query: str = Query(..., description="Natural language book description"),
    category: str = Query("All"),
    tone: str = Query("All"),
    top_k: int = Query(15, ge=1, le=50)
):
    if not query.strip():
        return []

    retrieved_isbns = pipeline.retrieve_semantic_recommendation(query)

    if not retrieved_isbns:
        return []

    df = pipeline.book_df[
        pipeline.book_df["isbn13"]
        .astype(str)
        .isin([str(isbn) for isbn in retrieved_isbns])
    ].copy()

    if category != "All":
        df = df[df["modified_category"] == category]

    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness",
    }

    if tone in tone_map and tone != "All":
        df.sort_values(by=tone_map[tone], ascending=False, inplace=True)

    df = df.head(top_k)

    results = []
    for _, row in df.iterrows():
        results.append(
            BookRecommendation(
                title=row["title"],
                authors=row.get("authors", "Unknown"),
                description=str(row.get("description_x", "")),
                thumbnail=row.get("large_thumbnail"),
                category=row.get("modified_category")
            )
        )

    return results
