---
date: 2026-09-01
---

# Improving the project

- Where possible, delegate the work to subagents
- Use opencode for coding tasks

## Context of this project

- This project was originally implemented as part of a machine learning competition
- There are no plans to put this in production. The main purpose is to expand it and learn about ML technologies.

## Starting from a clean base

Ensure all planned changes build on a solid foundation:

- [ ] Install all missing packages and also add them to requirements.txt (or the Dockerfile if they're system packages)
- [ ] Check for and fix existing bugs in this project
- [ ] Check for and fix anything that would add unnecessary friction during the implementation of the changes below

After these three steps, stop and summarize the changes. Then wait for my approval to continue with the remaining implementation because I want to check if the foundation is solid enough.

## Adding baselines

- [ ] Non-deep baseline
  - Model: lightgbm
  - Features: tf-idf features of the tweets, no metadata, implement ngram range as a tunable hyperparameter
  - Training: Use SMOTE on the tf-idf scores to address class imbalance
  - Evaluation:
    - Macro F1 score for classification
    - RMSE for regression
    - Holdout set (to stay consistent with the deep learning models where 5-fold cv is too compute intensive)
- [ ] LLM baseline
  - Use Microsoft Foundry with openai. Instructions [here](<https://learn.microsoft.com/en-us/training/modules/get-started-text-analysis-azure/3-language-sdk?pivots=text>)
  - Use the tiktoken library to estimate the number of tokens for the tweets in the validation dataset
    - Load as `tiktoken.get_encoding("o200k_base")` or `tiktoken.encoding_for_model("gpt-5")`
    - README for reference: <https://github.com/openai/tiktoken/blob/main/README.md>
  - Use `gpt-5.6-luna` (does exist, was released July 2026)
  - Use few-shot prompting, a system prompt for sentiment and get the classification- and regression label together in one prompt
  - Because there's no training involved in this, only use the validation dataset

## Infrastructure

- [ ] Make a plan for Azure ML infrastructure in terraform. I want to:

  - Store the dataset in Azure
  - Train and store the models on Azure
  - Use mlflow
  - At a later point, try creating an inference endpoint (not for production use, just to try it out)

  Please write the plan as a markdown file in `./docs/`. Don't create any Azure resources or produce any side-effects, only make the plan.

## Model choice

- [ ] Check huggingface hub (<https://huggingface.co/models>) for BERT encoders that are likely to perform better on the metrics used in this project. Only pick among models that can be fine tuned with <=24GB VRAM.
- [ ] Create a top-3 from the encoders you found and the encoders already used in the project based on what you expect to perform best on the metrics (Macro F1 and RMSE)
- [ ] Write your encoder selection + short justification to a markdown file in ./docs
