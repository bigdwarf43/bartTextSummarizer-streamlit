
def __chunk_text(  text):
    sentences = [ s + ' ' for s in sentence_segmentation(text, minimum_n_words_to_accept_sentence=1, language= "English") ]

    chunks = []

    chunk = ''

    length = 0

    for sentence in sentences:
        tokenized_sentence =  tokenizer.encode(sentence, truncation=False, max_length=None, return_tensors='pt') [0]

        if len(tokenized_sentence) >  tokenizer.model_max_length:
            continue

        length += len(tokenized_sentence)

        if length <=  tokenizer.model_max_length:
            chunk = chunk + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
            length = len(tokenized_sentence)

    if len(chunk) > 0:
        chunks.append(chunk.strip())

    return chunks

def __clean_text( text):
    if text.count('.') == 0:
        return text.strip()

    end_index = text.rindex('.') + 1

    return text[0 : end_index].strip()

def approach4( text):
    _device = 'cuda'
    global tokenizer, model
    chunk_texts =  __chunk_text(text)

    chunk_summaries = []

    for chunk_text in chunk_texts:
        input_tokenized =  tokenizer.encode(chunk_text, return_tensors='pt')

        input_tokenized = input_tokenized.to( _device)

        summary_ids =  model.to( _device).generate(input_tokenized, length_penalty=3.0, min_length = int(0.2 * len(chunk_text)), max_length = int(0.3 * len(chunk_text)), early_stopping=True, num_beams=5, no_repeat_ngram_size=2)

        output = [ tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids]

        chunk_summaries.append(output)

    # summaries = [  __clean_text(text) for chunk_summary in chunk_summaries for text in chunk_summary ]
    
    return chunk_summaries

def sentence_segmentation(document, minimum_n_words_to_accept_sentence, language):
    paragraphs = list(filter(lambda o: len(o.strip()) > 0, document.split('\n')))

    paragraphs = [ p.strip() for p in paragraphs ]

    paragraph_sentences = [nltk.sent_tokenize(p, language=language) for p in paragraphs ]

    paragraph_sentences = nltk.chain(*paragraph_sentences)

    paragraph_sentences = [ s.strip() for s in paragraph_sentences ]

    normal_word_tokenizer = nltk.tokenize.RegexpTokenizer(r'[^\W_]+')

    paragraph_sentences = filter(lambda o: len(normal_word_tokenizer.tokenize(o)) >= minimum_n_words_to_accept_sentence, paragraph_sentences)

    return list(paragraph_sentences)
