import gradio as gr
from prediction import PredictionPipeline
import os
obj = PredictionPipeline()


def get_recommendation(query: str, category: str, tone: str, top: int = 15):
    if not query.strip():
        return []

    retrieved_isbns = obj.retrieve_semantic_recommendation(query)

    if not retrieved_isbns:
        return []

    book_recs = obj.book_df[obj.book_df["isbn13"].astype(str).isin([str(isbn) for isbn in retrieved_isbns])].copy()

    if category != "All":
        book_recs = book_recs[book_recs["modified_category"] == category]

    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness",
    }

    if tone in tone_map:
        book_recs.sort_values(by=tone_map[tone],ascending=False,inplace=True)

    return book_recs.head(top)


def display(query: str, category: str, tone: str):
    recommendations = get_recommendation(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = str(row.get("description_x", ""))
        truncated_desc = " ".join(description.split()[:30]) + "..."

        authors_raw = str(row.get("authors", "Unknown author"))
        authors_split = authors_raw.split(";")

        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors_raw

        caption = f"**{row['title']}**  \n{authors_str}  \n{truncated_desc}"

        results.append(
            (row.get("large_thumbnail", None), caption)
        )

    return results


categories = ["All"] + sorted(
    obj.book_df["modified_category"].dropna().unique().tolist()
)

tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Semantic Book Recommender"
) as dashboard:

    gr.Markdown(
        """
        # 📚 Semantic Book Recommender  
        Describe a book you feel like reading.  
        The system understands *meaning*, not keywords.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            user_query = gr.Textbox(label="Book description",placeholder="A quiet story about nature and solitude",lines=2)

            submit_button = gr.Button("Find recommendations",variant="primary")

        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(choices=categories,label="Category",value="All")

            tone_dropdown = gr.Dropdown(choices=tones,label="Emotional tone",value="All")

    gr.Markdown("## Recommendations")

    output = gr.Gallery(label="",columns=6,rows=2,height="auto",show_label=False)

    submit_button.click(fn=display,inputs=[user_query, category_dropdown, tone_dropdown],outputs=output)


if __name__ == "__main__":
    dashboard.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 8080)), share=False)

