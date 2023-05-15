import nltk

from sumy.parsers import DocumentParser
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.summarizers.lsa import LsaSummarizer

from sumy.utils import get_stop_words
from transformers import Pipeline


class Summarizer:
    DEFAULT_LANGUAGE = "english"
    DEFAULT_EXTRACTED_ARTICLE_SENTENCES_LENGTH = 15

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        stemmer = Stemmer(Summarizer.DEFAULT_LANGUAGE)
        self.lsa_summarizer = LsaSummarizer(stemmer)
        self.lsa_summarizer.stop_words = get_stop_words(language=Summarizer.DEFAULT_LANGUAGE)

    @staticmethod
    def sentence_list(summarized_sentences) -> list:
        summarized_list = []
        for sentence in summarized_sentences:
            summarized_list.append(sentence._text)
        return summarized_list

    @staticmethod
    def join_sentences(summary_sentences: list) -> str:
        return " ".join([sentence for sentence in summary_sentences])

    @staticmethod
    def split_sentences_by_token_length(summary_sentences: list, split_token_length: int) -> list:
        accumulated_lists = []
        result_list = []
        cumulative_token_length = 0
        for sentence in summary_sentences:
            token_list = [token for token in nltk.word_tokenize(sentence) if token not in ['.']]
            token_length = len(token_list)
            if token_length + cumulative_token_length > split_token_length and result_list:
                accumulated_lists.append(Summarizer.join_sentences(result_list))
                result_list = [sentence]
                cumulative_token_length = token_length
            else:
                result_list.append(sentence)
                cumulative_token_length += token_length
        if result_list:
            accumulated_lists.append(Summarizer.join_sentences(result_list))
        return accumulated_lists

    def __extractive_summary(self, parser: DocumentParser, sentences_count) -> list:
        summarized_sentences = self.lsa_summarizer(parser.document, sentences_count)
        summarized_list = Summarizer.sentence_list(summarized_sentences)
        return summarized_list

    def extractive_summary_from_text(self, text: str, sentences_count: int) -> list:
        parser = PlaintextParser.from_string(text, Tokenizer(Summarizer.DEFAULT_LANGUAGE))
        return self.__extractive_summary(parser, sentences_count)

    def extractive_summary_from_url(self, url: str, sentences_count: int) -> list:
        parser = HtmlParser.from_url(url, Tokenizer(Summarizer.DEFAULT_LANGUAGE))
        return self.__extractive_summary(parser, sentences_count)

    def abstractive_summary(self, extract_summary_sentences: list) -> list:
        """
        :param extract_summary_sentences: Extractive summary of sentences after Latent semantic analysis
        :return: List of abstractive summary of sentences after calling distilbart-tos-summarizer-tosdr tokenizer
        """
        wrapped_sentences = Summarizer.split_sentences_by_token_length(extract_summary_sentences,
                                                                       split_token_length=600)
        abstractive_summary_list = []
        for result in self.pipeline(wrapped_sentences, min_length=32, max_length=512):
            abstractive_summary_list.append(result['summary_text'])
        return abstractive_summary_list