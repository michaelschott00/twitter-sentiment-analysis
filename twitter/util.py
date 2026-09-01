"""Implements several utilities, such as loss functions, pooling and data normalization and automates creating the files for the submissions."""

import os
from typing import Any, Literal

import emoji
import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter
from nltk.tokenize import TweetTokenizer
from torch import nn


class SCLoss(nn.Module):
    """Mostly taken from https://github.com/google-research/google-research/blob/master/supcon/losses.py#L240"""

    def __init__(self, temperature: float = 1.0, reduction: Literal["mean", "none"] = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, z, labels):
        logits = torch.matmul(z, z.T)  # pairwise dot products
        logits /= self.temperature
        label_mask = torch.matmul(labels.float(), labels.float().T).bool()

        # for numerical stability, subtract max
        with torch.no_grad():
            logits -= torch.max(logits, dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits)

        positive_mask = label_mask * (~torch.eye(len(labels), dtype=torch.bool, device=z.device))
        negative_mask = (~positive_mask) * (~torch.eye(len(labels), dtype=torch.bool, device=z.device))

        denominator = (exp_logits * positive_mask).sum(dim=1, keepdim=True) + (exp_logits * negative_mask).sum(dim=1, keepdim=True)
        loss = (logits - torch.log(denominator)) * positive_mask
        loss = loss.sum(dim=1)
        loss = torch.nan_to_num(loss / positive_mask.sum(dim=1))
        loss = -loss
        loss *= self.temperature  # counteract scaling the gradient
        if self.reduction == "mean":
            loss = loss.mean(dim=0)
        else:
            loss = loss[:len(labels) // 2]
        return loss


class MeanPooling(nn.Module):

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class SubmissionWriter(BasePredictionWriter):
    TEAM_ID = 16
    PREDICTION_DIR = "predictions"

    def __init__(self):
        super().__init__("epoch")

    def write_on_epoch_end(self, trainer, pl_module: 'LightningModule', predictions: list[Any], batch_indices: list[Any]):
        for dataloader, dl_predictions in zip(trainer.predict_dataloaders, predictions):
            dataset = dataloader.dataset
            if pl_module.hparams.task == "clf":
                final_predictions = np.array([dataset.INVERSE_LABEL_CODING[label.item()] for label in torch.cat(dl_predictions)])
                assert isinstance(final_predictions[0], str) or isinstance(final_predictions[0], np.str_), f"Predictions should be strings, but are {type(final_predictions[0])}"
            else:
                final_predictions = np.array([label.item() for label in torch.cat(dl_predictions)])
            assert final_predictions.shape[0] == 1000, f"Predictions should have 1000 elements, but have {final_predictions.shape[0]}"
            filename = os.path.join(self.PREDICTION_DIR, f'{self.TEAM_ID}__{dataset.split}__{pl_module.hparams.task}_pred.npy')
            os.makedirs(self.PREDICTION_DIR, exist_ok=True)
            np.save(filename, final_predictions)


class FeatureWriter(BasePredictionWriter):

    def __init__(self):
        super().__init__("epoch")

    def write_on_epoch_end(self, trainer, pl_module: 'LightningModule', predictions: list[Any], batch_indices: list[Any]):
        for dataloader, dl_predictions in zip(trainer.predict_dataloaders, predictions):
            dataset = dataloader.dataset
            csv_name = os.path.join(dataset.root_dir, f"tweets_{dataset.split}")

            # add to eixting frame if possible
            if os.path.exists(csv_name + "_bert.csv"):
                csv_name = csv_name + "_bert"
            df = pd.read_csv(csv_name + ".csv")
            if pl_module.hparams.task == "reg":
                df.loc[:, "bert_score"] = torch.cat(dl_predictions).numpy()
            elif pl_module.hparams.task == "clf":
                df.loc[:, "bert_sentiment"] = torch.cat(dl_predictions).numpy()

            if "bert" not in csv_name:
                csv_name = csv_name + "_bert"

            df.to_csv(csv_name + ".csv", index=False)


tokenizer = TweetTokenizer()


class TweetNormalizer(nn.Module):

    def __init__(self, lowercase: bool = True, replace_emoticons: bool = False):
        super().__init__()
        self.lowercase = lowercase
        self.replace_emoticons = replace_emoticons

    def normalizeToken(self, token):
        if token.startswith("@") and len(token) > 1:
            return "@USER" if not self.lowercase else "@user"
        elif token.lower().startswith("http") or token.lower().startswith("www"):
            return "HTTPURL" if not self.lowercase else "http"
        elif emoji.is_emoji(token):
            return emoji.demojize(token)
        elif emoji.purely_emoji(token):
            return ' '.join([emoji.demojize(c) for c in token])
        elif token in EMOTICONS and self.replace_emoticons:
            return EMOTICONS[token]
        elif token == "&amp;" or token == "&":
            return "and"
        elif token == "&lt;":
            return "<"
        elif token == "&gt;":
            return ">"
        else:
            return token

    def forward(self, tweet):
        tokens = tokenizer.tokenize(tweet)
        normTweet = " ".join([self.normalizeToken(token) for token in tokens])
        return " ".join(normTweet.split())


class WordsToSentence(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: str) -> str:
        import ast
        return " ".join(ast.literal_eval(x))


def first_n_examples(batch, n):
    input_ids = batch["input_ids"][:min(len(batch["input_ids"]), n)].detach().cpu().clone()
    attention_mask = batch["attention_mask"][:min(len(batch["attention_mask"]), n)].detach().cpu().clone()
    X = {"input_ids": input_ids, "attention_mask": attention_mask}
    y = batch["labels"][:min(len(batch["labels"]), n)].detach().cpu().clone()
    return X, y


EMOTICONS = {
    ":‑)": "Happy face or smiley",
    ":-))": "Very Happy face or smiley",
    ":-)))": "Very very Happy face or smiley",
    ":)": "Happy face or smiley",
    ":))": "Very Happy face or smiley",
    ":)))": "Very very Happy face or smiley",
    ":-]": "Happy face or smiley",
    ":]": "Happy face or smiley",
    ":-3": "Happy face smiley",
    ":3": "Happy face smiley",
    ":->": "Happy face smiley",
    ":>": "Happy face smiley",
    "8-)": "Happy face smiley",
    ":o)": "Happy face smiley",
    ":-}": "Happy face smiley",
    ":}": "Happy face smiley",
    ":-)": "Happy face smiley",
    ":c)": "Happy face smiley",
    ":^)": "Happy face smiley",
    "=]": "Happy face smiley",
    "=)": "Happy face smiley",
    ":‑D": "Laughing, big grin or laugh with glasses",
    ":D": "Laughing, big grin or laugh with glasses",
    "8‑D": "Laughing, big grin or laugh with glasses",
    "8D": "Laughing, big grin or laugh with glasses",
    "X‑D": "Laughing, big grin or laugh with glasses",
    "XD": "Laughing, big grin or laugh with glasses",
    "=D": "Laughing, big grin or laugh with glasses",
    "=3": "Laughing, big grin or laugh with glasses",
    "B^D": "Laughing, big grin or laugh with glasses",
    ":-(": "Frown, sad, angry or pouting",
    ":‑(": "Frown, sad, angry or pouting",
    ":(": "Frown, sad, angry or pouting",
    ":‑c": "Frown, sad, angry or pouting",
    ":c": "Frown, sad, angry or pouting",
    ":‑<": "Frown, sad, angry or pouting",
    ":<": "Frown, sad, angry or pouting",
    ":‑[": "Frown, sad, angry or pouting",
    ":[": "Frown, sad, angry or pouting",
    ":-||": "Frown, sad, angry or pouting",
    ">:[": "Frown, sad, angry or pouting",
    ":{": "Frown, sad, angry or pouting",
    ":@": "Frown, sad, angry or pouting",
    ">:(": "Frown, sad, angry or pouting",
    ":'‑(": "Crying",
    ":'(": "Crying",
    ":'‑)": "Tears of happiness",
    ":')": "Tears of happiness",
    "D‑':": "Horror",
    "D:<": "Disgust",
    "D:": "Sadness",
    "D8": "Great dismay",
    "D;": "Great dismay",
    "D=": "Great dismay",
    "DX": "Great dismay",
    ":‑O": "Surprise",
    ":O": "Surprise",
    ":‑o": "Surprise",
    ":o": "Surprise",
    ":-0": "Shock",
    "8‑0": "Yawn",
    ">:O": "Yawn",
    ":-*": "Kiss",
    ":*": "Kiss",
    ":X": "Kiss",
    ";‑)": "Wink or smirk",
    ";)": "Wink or smirk",
    "*-)": "Wink or smirk",
    "*)": "Wink or smirk",
    ";‑]": "Wink or smirk",
    ";]": "Wink or smirk",
    ";^)": "Wink or smirk",
    ":‑,": "Wink or smirk",
    ";D": "Wink or smirk",
    ":‑P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    "X‑P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    "XP": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":‑Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    "d:": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    "=p": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    ">:P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":‑/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    ":/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    ":-[.]": "Skeptical, annoyed, undecided, uneasy or hesitant",
    '>:[(\\)]': "Skeptical, annoyed, undecided, uneasy or hesitant",
    ">:/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    ":[(\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
    "=/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    "=[(\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
    ":L": "Skeptical, annoyed, undecided, uneasy or hesitant",
    "=L": "Skeptical, annoyed, undecided, uneasy or hesitant",
    ":S": "Skeptical, annoyed, undecided, uneasy or hesitant",
    ":‑|": "Straight face",
    ":|": "Straight face",
    ":$": "Embarrassed or blushing",
    ":‑x": "Sealed lips or wearing braces or tongue-tied",
    ":x": "Sealed lips or wearing braces or tongue-tied",
    ":‑#": "Sealed lips or wearing braces or tongue-tied",
    ":#": "Sealed lips or wearing braces or tongue-tied",
    ":‑&": "Sealed lips or wearing braces or tongue-tied",
    ":&": "Sealed lips or wearing braces or tongue-tied",
    "O:‑)": "Angel, saint or innocent",
    "O:)": "Angel, saint or innocent",
    "0:‑3": "Angel, saint or innocent",
    "0:3": "Angel, saint or innocent",
    "0:‑)": "Angel, saint or innocent",
    "0:)": "Angel, saint or innocent",
    ":‑b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    "0;^)": "Angel, saint or innocent",
    ">:‑)": "Evil or devilish",
    ">:)": "Evil or devilish",
    "}:‑)": "Evil or devilish",
    "}:)": "Evil or devilish",
    "3:‑)": "Evil or devilish",
    "3:)": "Evil or devilish",
    ">;)": "Evil or devilish",
    "|;‑)": "Cool",
    "|‑O": "Bored",
    ":‑J": "Tongue-in-cheek",
    "#‑)": "Party all night",
    "%‑)": "Drunk or confused",
    "%)": "Drunk or confused",
    ":-###..": "Being sick",
    ":###..": "Being sick",
    "<:‑|": "Dump",
    "(>_<)": "Troubled",
    "(>_<)>": "Troubled",
    "(';')": "Baby",
    "(^^>``": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "(^_^;)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "(-_-;)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "(~_~;) (・.・;)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "(-_-)zzz": "Sleeping",
    "(^_-)": "Wink",
    "((+_+))": "Confused",
    "(+o+)": "Confused",
    "(o|o)": "Ultraman",
    "^_^": "Joyful",
    "(^_^)/": "Joyful",
    "(^O^)／": "Joyful",
    "(^o^)／": "Joyful",
    "(__)": "Kowtow as a sign of respect, or dogeza for apology",
    "_(._.)_": "Kowtow as a sign of respect, or dogeza for apology",
    "<(_ _)>": "Kowtow as a sign of respect, or dogeza for apology",
    "<m(__)m>": "Kowtow as a sign of respect, or dogeza for apology",
    "m(__)m": "Kowtow as a sign of respect, or dogeza for apology",
    "m(_ _)m": "Kowtow as a sign of respect, or dogeza for apology",
    "('_')": "Sad or Crying",
    "(/_;)": "Sad or Crying",
    "(T_T) (;_;)": "Sad or Crying",
    "(;_;": "Sad of Crying",
    "(;_:)": "Sad or Crying",
    "(;O;)": "Sad or Crying",
    "(:_;)": "Sad or Crying",
    "(ToT)": "Sad or Crying",
    ";_;": "Sad or Crying",
    ";-;": "Sad or Crying",
    ";n;": "Sad or Crying",
    ";;": "Sad or Crying",
    "Q.Q": "Sad or Crying",
    "T.T": "Sad or Crying",
    "QQ": "Sad or Crying",
    "Q_Q": "Sad or Crying",
    "(-.-)": "Shame",
    "(-_-)": "Shame",
    "(一一)": "Shame",
    "(；一_一)": "Shame",
    "(=_=)": "Tired",
    "(=^·^=)": "cat",
    "(=^··^=)": "cat",
    "=_^= ": "cat",
    "(..)": "Looking down",
    "(._.)": "Looking down",
    "^m^": "Giggling with hand covering mouth",
    "(・・?": "Confusion",
    "(?_?)": "Confusion",
    ">^_^<": "Normal Laugh",
    "<^!^>": "Normal Laugh",
    "^/^": "Normal Laugh",
    "（*^_^*）": "Normal Laugh",
    "(^<^) (^.^)": "Normal Laugh",
    "(^^)": "Normal Laugh",
    "(^.^)": "Normal Laugh",
    "(^_^.)": "Normal Laugh",
    "(^_^)": "Normal Laugh",
    "(^^)": "Normal Laugh",
    "(^J^)": "Normal Laugh",
    "(*^.^*)": "Normal Laugh",
    "(^—^）": "Normal Laugh",
    "(#^.^#)": "Normal Laugh",
    "（^—^）": "Waving",
    "(;_;)/~~~": "Waving",
    "(^.^)/~~~": "Waving",
    "(-_-)/~~~ ($··)/~~~": "Waving",
    "(T_T)/~~~": "Waving",
    "(ToT)/~~~": "Waving",
    "(*^0^*)": "Excited",
    "(*_*)": "Amazed",
    "(*_*;": "Amazed",
    "(+_+) (@_@)": "Amazed",
    "(*^^)v": "Laughing,Cheerful",
    "(^_^)v": "Laughing,Cheerful",
    "((d[-_-]b))": "Headphones,Listening to music",
    '(-"-)': "Worried",
    "(ーー;)": "Worried",
    "(^0_0^)": "Eyeglasses",
    "(＾ｖ＾)": "Happy",
    "(＾ｕ＾)": "Happy",
    "(^)o(^)": "Happy",
    "(^O^)": "Happy",
    "(^o^)": "Happy",
    ")^o^(": "Happy",
    ":O o_O": "Surprised",
    "o_0": "Surprised",
    "o.O": "Surpised",
    "(o.o)": "Surprised",
    "oO": "Surprised",
    "(*￣m￣)": "Dissatisfied",
    "(‘A`)": "Snubbed or Deflated"
}
