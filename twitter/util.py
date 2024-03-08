"""Implements several utilities, such as loss functions, pooling and data normalization and automates creating the files for the submissions."""

import torch
from torch import nn
import emoji
from nltk.tokenize import TweetTokenizer
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import BasePredictionWriter
import os
from typing import List, Any, Literal
import numpy as np
import pandas as pd


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

    def write_on_epoch_end(self, trainer, pl_module: 'LightningModule', predictions: List[Any], batch_indices: List[Any]):
        for dataloader, dl_predictions in zip(trainer.predict_dataloaders, predictions):
            dataset = dataloader.dataset
            if pl_module.hparams.task == "clf":
                final_predictions = np.array([dataset.INVERSE_LABEL_CODING[label.item()] for label in torch.cat(dl_predictions)])
                assert (isinstance(final_predictions[0], str)) or (isinstance(final_predictions[0]), np.str_), f"Predictions should be strings, but are {type(final_predictions[0])}"
            else:
                final_predictions = np.array([label.item() for label in torch.cat(dl_predictions)])
            assert final_predictions.shape[0] == 1000, f"Predictions should have 1000 elements, but have {final_predictions.shape[0]}"
            filename = os.path.join(self.PREDICTION_DIR, f'{self.TEAM_ID}__{dataset.split}__{pl_module.hparams.task}_pred.npy')
            np.save(filename, final_predictions)


class FeatureWriter(BasePredictionWriter):

    def __init__(self):
        super().__init__("epoch")

    def write_on_epoch_end(self, trainer, pl_module: 'LightningModule', predictions: List[Any], batch_indices: List[Any]):
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
        return " ".join(eval(x))


def first_n_examples(batch, n):
    input_ids = batch["input_ids"][:min(len(batch["input_ids"]), n)].detach().cpu().clone()
    attention_mask = batch["attention_mask"][:min(len(batch["attention_mask"]), n)].detach().cpu().clone()
    X = {"input_ids": input_ids, "attention_mask": attention_mask}
    y = batch["labels"][:min(len(batch["labels"]), n)].detach().cpu().clone()
    return X, y


EMOTICONS = {
    u":‑)": "Happy face or smiley",
    u":-))": "Very Happy face or smiley",
    u":-)))": "Very very Happy face or smiley",
    u":)": "Happy face or smiley",
    u":))": "Very Happy face or smiley",
    u":)))": "Very very Happy face or smiley",
    u":-]": "Happy face or smiley",
    u":]": "Happy face or smiley",
    u":-3": "Happy face smiley",
    u":3": "Happy face smiley",
    u":->": "Happy face smiley",
    u":>": "Happy face smiley",
    u"8-)": "Happy face smiley",
    u":o)": "Happy face smiley",
    u":-}": "Happy face smiley",
    u":}": "Happy face smiley",
    u":-)": "Happy face smiley",
    u":c)": "Happy face smiley",
    u":^)": "Happy face smiley",
    u"=]": "Happy face smiley",
    u"=)": "Happy face smiley",
    u":‑D": "Laughing, big grin or laugh with glasses",
    u":D": "Laughing, big grin or laugh with glasses",
    u"8‑D": "Laughing, big grin or laugh with glasses",
    u"8D": "Laughing, big grin or laugh with glasses",
    u"X‑D": "Laughing, big grin or laugh with glasses",
    u"XD": "Laughing, big grin or laugh with glasses",
    u"=D": "Laughing, big grin or laugh with glasses",
    u"=3": "Laughing, big grin or laugh with glasses",
    u"B^D": "Laughing, big grin or laugh with glasses",
    u":-(": "Frown, sad, angry or pouting",
    u":‑(": "Frown, sad, angry or pouting",
    u":(": "Frown, sad, angry or pouting",
    u":‑c": "Frown, sad, angry or pouting",
    u":c": "Frown, sad, angry or pouting",
    u":‑<": "Frown, sad, angry or pouting",
    u":<": "Frown, sad, angry or pouting",
    u":‑[": "Frown, sad, angry or pouting",
    u":[": "Frown, sad, angry or pouting",
    u":-||": "Frown, sad, angry or pouting",
    u">:[": "Frown, sad, angry or pouting",
    u":{": "Frown, sad, angry or pouting",
    u":@": "Frown, sad, angry or pouting",
    u">:(": "Frown, sad, angry or pouting",
    u":'‑(": "Crying",
    u":'(": "Crying",
    u":'‑)": "Tears of happiness",
    u":')": "Tears of happiness",
    u"D‑':": "Horror",
    u"D:<": "Disgust",
    u"D:": "Sadness",
    u"D8": "Great dismay",
    u"D;": "Great dismay",
    u"D=": "Great dismay",
    u"DX": "Great dismay",
    u":‑O": "Surprise",
    u":O": "Surprise",
    u":‑o": "Surprise",
    u":o": "Surprise",
    u":-0": "Shock",
    u"8‑0": "Yawn",
    u">:O": "Yawn",
    u":-*": "Kiss",
    u":*": "Kiss",
    u":X": "Kiss",
    u";‑)": "Wink or smirk",
    u";)": "Wink or smirk",
    u"*-)": "Wink or smirk",
    u"*)": "Wink or smirk",
    u";‑]": "Wink or smirk",
    u";]": "Wink or smirk",
    u";^)": "Wink or smirk",
    u":‑,": "Wink or smirk",
    u";D": "Wink or smirk",
    u":‑P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"X‑P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"XP": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"d:": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"=p": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u">:P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u":‑/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":-[.]": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u'>:[(\\)]': "Skeptical, annoyed, undecided, uneasy or hesitant",
    u">:/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":[(\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=[(\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":L": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u"=L": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":S": "Skeptical, annoyed, undecided, uneasy or hesitant",
    u":‑|": "Straight face",
    u":|": "Straight face",
    u":$": "Embarrassed or blushing",
    u":‑x": "Sealed lips or wearing braces or tongue-tied",
    u":x": "Sealed lips or wearing braces or tongue-tied",
    u":‑#": "Sealed lips or wearing braces or tongue-tied",
    u":#": "Sealed lips or wearing braces or tongue-tied",
    u":‑&": "Sealed lips or wearing braces or tongue-tied",
    u":&": "Sealed lips or wearing braces or tongue-tied",
    u"O:‑)": "Angel, saint or innocent",
    u"O:)": "Angel, saint or innocent",
    u"0:‑3": "Angel, saint or innocent",
    u"0:3": "Angel, saint or innocent",
    u"0:‑)": "Angel, saint or innocent",
    u"0:)": "Angel, saint or innocent",
    u":‑b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    u"0;^)": "Angel, saint or innocent",
    u">:‑)": "Evil or devilish",
    u">:)": "Evil or devilish",
    u"}:‑)": "Evil or devilish",
    u"}:)": "Evil or devilish",
    u"3:‑)": "Evil or devilish",
    u"3:)": "Evil or devilish",
    u">;)": "Evil or devilish",
    u"|;‑)": "Cool",
    u"|‑O": "Bored",
    u":‑J": "Tongue-in-cheek",
    u"#‑)": "Party all night",
    u"%‑)": "Drunk or confused",
    u"%)": "Drunk or confused",
    u":-###..": "Being sick",
    u":###..": "Being sick",
    u"<:‑|": "Dump",
    u"(>_<)": "Troubled",
    u"(>_<)>": "Troubled",
    u"(';')": "Baby",
    u"(^^>``": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"(^_^;)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"(-_-;)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"(~_~;) (・.・;)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    u"(-_-)zzz": "Sleeping",
    u"(^_-)": "Wink",
    u"((+_+))": "Confused",
    u"(+o+)": "Confused",
    u"(o|o)": "Ultraman",
    u"^_^": "Joyful",
    u"(^_^)/": "Joyful",
    u"(^O^)／": "Joyful",
    u"(^o^)／": "Joyful",
    u"(__)": "Kowtow as a sign of respect, or dogeza for apology",
    u"_(._.)_": "Kowtow as a sign of respect, or dogeza for apology",
    u"<(_ _)>": "Kowtow as a sign of respect, or dogeza for apology",
    u"<m(__)m>": "Kowtow as a sign of respect, or dogeza for apology",
    u"m(__)m": "Kowtow as a sign of respect, or dogeza for apology",
    u"m(_ _)m": "Kowtow as a sign of respect, or dogeza for apology",
    u"('_')": "Sad or Crying",
    u"(/_;)": "Sad or Crying",
    u"(T_T) (;_;)": "Sad or Crying",
    u"(;_;": "Sad of Crying",
    u"(;_:)": "Sad or Crying",
    u"(;O;)": "Sad or Crying",
    u"(:_;)": "Sad or Crying",
    u"(ToT)": "Sad or Crying",
    u";_;": "Sad or Crying",
    u";-;": "Sad or Crying",
    u";n;": "Sad or Crying",
    u";;": "Sad or Crying",
    u"Q.Q": "Sad or Crying",
    u"T.T": "Sad or Crying",
    u"QQ": "Sad or Crying",
    u"Q_Q": "Sad or Crying",
    u"(-.-)": "Shame",
    u"(-_-)": "Shame",
    u"(一一)": "Shame",
    u"(；一_一)": "Shame",
    u"(=_=)": "Tired",
    u"(=^·^=)": "cat",
    u"(=^··^=)": "cat",
    u"=_^= ": "cat",
    u"(..)": "Looking down",
    u"(._.)": "Looking down",
    u"^m^": "Giggling with hand covering mouth",
    u"(・・?": "Confusion",
    u"(?_?)": "Confusion",
    u">^_^<": "Normal Laugh",
    u"<^!^>": "Normal Laugh",
    u"^/^": "Normal Laugh",
    u"（*^_^*）": "Normal Laugh",
    u"(^<^) (^.^)": "Normal Laugh",
    u"(^^)": "Normal Laugh",
    u"(^.^)": "Normal Laugh",
    u"(^_^.)": "Normal Laugh",
    u"(^_^)": "Normal Laugh",
    u"(^^)": "Normal Laugh",
    u"(^J^)": "Normal Laugh",
    u"(*^.^*)": "Normal Laugh",
    u"(^—^）": "Normal Laugh",
    u"(#^.^#)": "Normal Laugh",
    u"（^—^）": "Waving",
    u"(;_;)/~~~": "Waving",
    u"(^.^)/~~~": "Waving",
    u"(-_-)/~~~ ($··)/~~~": "Waving",
    u"(T_T)/~~~": "Waving",
    u"(ToT)/~~~": "Waving",
    u"(*^0^*)": "Excited",
    u"(*_*)": "Amazed",
    u"(*_*;": "Amazed",
    u"(+_+) (@_@)": "Amazed",
    u"(*^^)v": "Laughing,Cheerful",
    u"(^_^)v": "Laughing,Cheerful",
    u"((d[-_-]b))": "Headphones,Listening to music",
    u'(-"-)': "Worried",
    u"(ーー;)": "Worried",
    u"(^0_0^)": "Eyeglasses",
    u"(＾ｖ＾)": "Happy",
    u"(＾ｕ＾)": "Happy",
    u"(^)o(^)": "Happy",
    u"(^O^)": "Happy",
    u"(^o^)": "Happy",
    u")^o^(": "Happy",
    u":O o_O": "Surprised",
    u"o_0": "Surprised",
    u"o.O": "Surpised",
    u"(o.o)": "Surprised",
    u"oO": "Surprised",
    u"(*￣m￣)": "Dissatisfied",
    u"(‘A`)": "Snubbed or Deflated"
}
