# Improving the project

- Where possible, delegate the work to subagents
- Use opencode for coding tasks

## Starting from a clean base

Ensure all planned changes build on a solid foundation:

- [ ] Check for and fix existing bugs in this project
- [ ] Check for and fix anything that would add unnecessary friction during the implementation of the changes below
- [ ] Remove anything that would not add value after the changes have been implemented. Also remove anything that's generally unnecessary in this project.

After these three steps, stop and summarize the changes.

## Adding baselines

- [ ] Pick one of lightgbm or xgboost, whichever is likely to perform better, and implement a baseline using tf-idf features
- [ ] Implement an LLM baseline
  - Use the tiktoken library to estimate the number of tokens for the tweets in the validation dataset. This enables a cost estimate.
  - Because there's no training, only use the validation dataset
  - Given that I plan to use Azure throughout, use the openai azure api
  - Pick GPT 5.6 Luna

## Infrastructure

- [ ] Evaluate where Azure ML could be used in this project. Write your findings to a markdown file in ./docs/

## Model choice

- [ ] Research suitable BERT encoders and pick the top 3 based on:
  - Similarity of the pre-training dataset
  - Ability to fine-tune on a single 24GB VRAM GPU (it should be)
  - Any additional factors you think are likely to lead to good performance
- [ ] Write your encoder selection + short justification to a markdown file in ./docs

## Training tweaks

- [ ] Check if SMOTE would improve over current techniques for dealing with class imbalance
- [ ] If after the previous analysis, you deem it worth it, add SMOTE to this project
