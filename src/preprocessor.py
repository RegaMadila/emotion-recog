import re
from transformers import BertTokenizer

class TextPreprocessor:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.happy_emoticons = [':)', ': )', ':-)', '=)', ':]', ';)', ';-)']
        self.sad_emoticons = [':(', ': (', ': [', '= [', ':[', ':-(', '=(', '; [', ';(']
        self.flat_emoticons = [':/', ':|']
        self.emoticon_pattern = re.compile(r'[:;xX=]-?[dDpP/\\]?\s*[\(\)\[\]\{\}|]')

    def replace_emoticons(self, text):
        if not isinstance(text, str):
            return ""
            
        for emo in self.happy_emoticons:
            text = text.replace(emo, '{happy_face}')
        for emo in self.sad_emoticons:
            text = text.replace(emo, '{sad_face}')
        for emo in self.flat_emoticons:
            text = text.replace(emo, '{flat_face}')
            
        text = re.sub(self.emoticon_pattern, '', text)
        return text

    def bert_tokenize(self, text):
        if not isinstance(text, str):
            return ""
        tokens = self.tokenizer.tokenize(text)
        return ' '.join(tokens)

    def preprocess(self, df, text_column='text'):
        df = df.copy()
        df['text_processed'] = df[text_column].apply(self.replace_emoticons)

        return df
