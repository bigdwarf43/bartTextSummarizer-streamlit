import streamlit as st
from transformers import pipeline
from summarizerClass import Summarizer
import nltk

def main():
    @st.cache(allow_output_mutation=True,
              suppress_st_warning=True,
              show_spinner=False)
    def create_pipeline():
        with st.spinner('Please wait for the model to load...'):
            terms_and_conditions_pipeline = pipeline(
                task='summarization',
                model="sshleifer/distilbart-xsum-12-3",
                tokenizer="sshleifer/distilbart-xsum-12-3"
            )
        return terms_and_conditions_pipeline
    
    def abstractive_summary_from_cache(summary_sentences: tuple) -> tuple:
        with st.spinner('Summarizing the text is in progress...'):
            return tuple(summarizer.abstractive_summary(list(summary_sentences)))
    
    summarizer: Summarizer = Summarizer(create_pipeline())


    if 'tc_text' not in st.session_state:
        st.session_state['tc_text'] = ''

    if 'sentences_length' not in st.session_state:
        st.session_state['sentences_length'] = Summarizer.DEFAULT_EXTRACTED_ARTICLE_SENTENCES_LENGTH

    if 'sample_choice' not in st.session_state:
        st.session_state['sample_choice'] = ''

    summarize_button = st.button(label='Summarize')

    tc_text_input = st.text_area(
        value=st.session_state.tc_text,
        label='Enter data',
        height=240
    )

    
    # generate chunks of text \ sentences <= 1024 tokens
    def nest_sentences(document):
        nested = []
        sent = []
        length = 0
        for sentence in nltk.sent_tokenize(document):
            length += len(sentence)
            if length < 1024:
                sent.append(sentence)
            else:
                nested.append(sent)
                sent = [sentence]
                length = len(sentence)

        if sent:
            nested.append(sent)
        return nested


    if summarize_button:
        sentences_length = 15
        abstractSummary = []
        extractiveSummary = []
        nested = nest_sentences(tc_text_input)

        for chunk in nested:
            extract_summary_sentences = summarizer.extractive_summary_from_text(chunk, sentences_length)
            extract_summary_sentences_tuple = tuple(extract_summary_sentences)
            abstract_summary_tuple = abstractive_summary_from_cache(extract_summary_sentences_tuple)
            abstract_summary_list = list(abstract_summary_tuple)
            abstractSummary.append(abstract_summary_list)
            extractiveSummary.append(extract_summary_sentences)

        st.success('\n'.join(' '.join(map(str,sl)) for sl in abstractSummary))
        st.success(extractiveSummary)


if __name__ == "__main__":
    main()