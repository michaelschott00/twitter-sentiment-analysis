import gradio as gr
import pandas as pd


df = pd.read_csv('twitter/data/tweets_train.csv')
sample = df.sample(1)


def update_tweet(correct, total):
    sample = df.sample(1)
    return (
        sample['text'].values[0], "", None, total + 1, f"Accuracy: {round((correct / total) * 100, 2)}%"
    )


def check_answer(inp, exp, correct):
    if len(inp) == 0:
        return "Please select a sentiment.", "", correct

    if len(inp) > 1:
        return "Please select only one sentiment.", "", correct

    if inp[0] == sample['sentiment'].values[0]:
        return "Correct!", sample['sentiment'].values[0], correct + 1

    else:
        return "Incorrect.", sample['sentiment'].values[0], correct


with gr.Blocks() as demo:
    # interface
    tweet = gr.Textbox(value=sample['text'].values[0], label="Tweet")
    inp = gr.CheckboxGroup(choices=list(df['sentiment'].unique()), label="Sentiment:")
    out = gr.Textbox(label="Result:", interactive=False)
    exp = gr.Textbox(label="Expected:", interactive=False)
    next = gr.Button(value="Next")
    total = gr.Number(1, label="Total:", visible=False)
    correct = gr.Number(0, label="Correct:", visible=False)
    accuracy = gr.Label("Accuracy: 0%")

    # callbacks
    inp.change(check_answer, inputs=[inp, exp, correct], outputs=[out, exp, correct])
    next.click(update_tweet, inputs=[correct, total], outputs=[tweet, exp, inp, total, accuracy])

demo.launch()
