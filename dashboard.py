import pandas as pd
import numpy as np
import gradio as gr
from dotenv import load_dotenv
from recommendation import retrieve_semantic_recommendation

load_dotenv()


df = pd.read_csv("book_with_emotion.csv")


df['large_thumbnail'] = df['thumbnail'] + "&fife=w800"
df['large_thumbnail'] = np.where(
    df['large_thumbnail'].isna(), 
    "cover-not-found.jpg", 
    df['large_thumbnail']
)

def get_recommendation(query: str, category: str = "All", tone: str = "All", top: int = 15):
    retrieved_isbns = retrieve_semantic_recommendation(query)
    
    book_recs = df[df['isbn13'].astype(str).isin([str(isbn) for isbn in retrieved_isbns])].copy()


    if category != 'All':
        book_recs = book_recs[book_recs['modified_category'] == category]

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":  
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)
    
    return book_recs.head(top)

def display(query: str, category: str, tone: str):
    recommendations = get_recommendation(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():

        description = str(row["description_x"])
        truncated_desc = " ".join(description.split()[:30]) + "..."

    
        authors_raw = str(row["authors"])
        authors_split = authors_raw.split(";")
        
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors_raw

        caption = f"{row['title']} by {authors_str}: {truncated_desc}"
        results.append((row["large_thumbnail"], caption))
    
    return results


categories = ["All"] + sorted(df["modified_category"].dropna().unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness"
        )
        category_dropdown = gr.Dropdown(
            choices=categories, 
            label="Select a category:", 
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones, 
            label="Select an emotional tone:", 
            value="All"
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(
        fn=display,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    dashboard.launch(share=True)