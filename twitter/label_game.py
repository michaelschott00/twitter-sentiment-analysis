import os

import gradio as gr
import pandas as pd

# robust path resolution: try splits, then raw, then legacy
CANDIDATE_PATHS = [
    "data/splits/tweets_train.csv",
    "data/raw/tweets_train.csv",
    "twitter/data/tweets_train.csv",
]
csv_path = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), CANDIDATE_PATHS[0])
df = (
    pd.read_csv(csv_path)
    if os.path.exists(csv_path)
    else pd.DataFrame({"text": ["No data found"], "sentiment": ["neutral"]})
)
sample = df.sample(1) if len(df) > 0 else df

# keep mutable state in dict to avoid global rebinding issues
_state = {"sample": sample}


def update_tweet(correct, total):
    new_sample = df.sample(1)
    _state["sample"] = new_sample
    return (
        new_sample["text"].values[0],
        "",
        None,
        total + 1,
        f"Accuracy: {round((correct / total) * 100, 2)}%"
        if total > 0
        else "Accuracy: 0%",
    )


def check_answer(inp, exp, correct):
    if len(inp) == 0:
        return "Please select a sentiment.", "", correct
    if len(inp) > 1:
        return "Please select only one sentiment.", "", correct
    cur = _state["sample"]
    if inp[0] == cur["sentiment"].values[0]:
        return "Correct!", cur["sentiment"].values[0], correct + 1
    else:
        return "Incorrect.", cur["sentiment"].values[0], correct


with gr.Blocks() as demo:
    # interface
    tweet = gr.Textbox(
        value=_state["sample"]["text"].values[0] if len(df) > 0 else "", label="Tweet"
    )
    inp = gr.CheckboxGroup(
        choices=list(df["sentiment"].unique())
        if "sentiment" in df.columns
        else ["positive", "neutral", "negative"],
        label="Sentiment:",
    )
    out = gr.Textbox(label="Result:", interactive=False)
    exp = gr.Textbox(label="Expected:", interactive=False)
    next = gr.Button(value="Next")
    total = gr.Number(1, label="Total:", visible=False)
    correct = gr.Number(0, label="Correct:", visible=False)
    accuracy = gr.Label("Accuracy: 0%")

    # callbacks
    inp.change(check_answer, inputs=[inp, exp, correct], outputs=[out, exp, correct])
    next.click(
        update_tweet,
        inputs=[correct, total],
        outputs=[tweet, exp, inp, total, accuracy],
    )

if __name__ == "__main__":
    demo.launch()
