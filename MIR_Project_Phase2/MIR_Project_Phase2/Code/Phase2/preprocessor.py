from Phase1.Prepare_text.PrepareText import EnglishPreprocessor
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')


class Preprocessor(EnglishPreprocessor):

    def set_stopwords(self):
        self.stop_words = stopwords.words('english')
