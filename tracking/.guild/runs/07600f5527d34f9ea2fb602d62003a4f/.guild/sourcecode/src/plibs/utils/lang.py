import fasttext
import os

class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = f"{os.getenv('PRJ_HOME')}/models/fasttext/lid.176.ftz"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text, top_n=2):
        predictions = self.model.predict(text, k=top_n) # returns top N matching languages
        return predictions

    def is_language(self, txt, lang_code='__label__en', min_ratio_wrt_first=0.4, top_n=2):
        lang = self.predict_lang(txt, top_n=top_n)
        prob_ratio = abs(lang[1][1]/lang[1][0]) if len(lang[1]) > 1 else 0.9999
        return (lang[0][0] == lang_code or (lang[0][1] == lang_code and (prob_ratio >= min_ratio_wrt_first ) ), prob_ratio) # first or second but at least 10% of the first probabilty