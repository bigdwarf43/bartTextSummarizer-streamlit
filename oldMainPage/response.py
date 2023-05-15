import streamlit as st
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer
)
import torch
import nltk
import re
from transformers import pipeline
import numpy as np

model = None
tokenizer = None
summarizer = None

@st.cache_data()
def loadModel():
    global model, tokenizer, summarizer
    # model_name = "Bigdwarf43/results"
    
    # model_path = "C:/Users/yashw/.cache/huggingface/hub/models--Bigdwarf43--results/snapshots/f4da98c25966a7260ed890a62a8461ac3a5a2b17/"
    model_path = "sshleifer/distilbart-xsum-12-3"

    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained(model_path)
    summarizer = pipeline(task="summarization", model=model, tokenizer= tokenizer)
    print("modelLoaded")

def summarizeTextWithoutChunking(input):
    global summarizer
    output = summarizer(input)
    return output

def approach1(input):
    # global summarizer
    # output = summarizer(input)
    # return output
    nestedSentences = nest_sentences(input)
    output = generate_summary(nestedSentences)
    return output


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

# generate summary on text with <= 1024 tokens
def generate_summary(nested_sentences):
  global tokenizer, model, summarizer
  device = 'cuda'
  summaries = []
  for nested in nested_sentences:
    input_tokenized = tokenizer.encode(' '.join(nested), truncation=True, return_tensors='pt')
    input_tokenized = input_tokenized.to(device)

    # dct = tokenizer.batch_encode_plus(nested,
    #                                       max_length=1024,
    #                                       truncation=True,
    #                                       padding='max_length',
    #                                       return_tensors="pt")
    summary_ids = model.to(device).generate(input_tokenized,
            num_beams=4,
            length_penalty=2.0,
            max_length=142,
            min_length=56,
            no_repeat_ngram_size=3,
            early_stopping=True,
            decoder_start_token_id=tokenizer.eos_token_id,)
    output = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    summaries.append(output)
  summaries = [sentence for sublist in summaries for sentence in sublist]
  return summaries

#approach 2

def approach2(inputText):
    global tokenizer, model
    # tokenize without truncation
    inputs_no_trunc = tokenizer(inputText, max_length=None, return_tensors='pt', truncation=False)

    # get batches of tokens corresponding to the exact model_max_length
    chunk_start = 0
    chunk_end = tokenizer.model_max_length  # == 1024 for Bart
    inputs_batch_lst = []
    while chunk_start <= len(inputs_no_trunc['input_ids'][0]):
        inputs_batch = inputs_no_trunc['input_ids'][0][chunk_start:chunk_end]  # get batch of n tokens
        inputs_batch = torch.unsqueeze(inputs_batch, 0)
        inputs_batch_lst.append(inputs_batch)
        chunk_start += 100  # == 1024 for Bart
        chunk_end += 100 # == 1024 for Bart

    # generate a summary on each batch
    summary_ids_lst = [model.generate(inputs, num_beams=4, max_length=100, early_stopping=True) for inputs in inputs_batch_lst]

    # decode the output and join into one string with one paragraph per summary batch
    summary_batch_lst = []
    for summary_id in summary_ids_lst:
        summary_batch = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_id]
        summary_batch_lst.append(summary_batch[0])
    summary_all = '\n'.join(summary_batch_lst)

    print(summary_all)
    return summary_all

# approach 3
def approach3(input):
    sentences = nltk.tokenize.sent_tokenize(input)

    # initialize
    length = 0
    chunk = ""
    chunks = []
    count = -1
    for sentence in sentences:
        count += 1
        combined_length = len(tokenizer.tokenize(sentence)) + length # add the no. of sentence tokens to the length counter

        if combined_length  <= tokenizer.max_len_single_sentence: # if it doesn't exceed
            chunk += sentence + " " # add the sentence to the chunk
            length = combined_length # update the length counter

            # if it is the last sentence
            if count == len(sentences) - 1:
                chunks.append(chunk) # save the chunk
            
        else: 
            chunks.append(chunk) # save the chunk
            # reset 
            length = 0 
            chunk = ""

            # take care of the overflow sentence
            chunk += sentence + " "
            length = len(tokenizer.tokenize(sentence))

    # inputs
    inputs = [tokenizer(chunk, return_tensors="pt") for chunk in chunks]

    summary = []
    # print summary
    for input in inputs:
        output = model.generate(**input)
        summary.append(tokenizer.decode(*output, skip_special_tokens=True))
    
    return summary


#Approach 4

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


#approach 5
def recursive_bart_summarization(document, max_length=1024, min_length=0, num_beams=4, repetition_penalty=2.0, length_penalty=2.0):
    device = 'cuda'
    global tokenizer, model
    # Divide the document into chunks
    chunks = [document[i:i+max_length] for i in range(0, len(document), max_length)]
    
    # Initialize variables for summary and previous context
    summary = ""
    prev_context = ""
    summaryList = []
    # Iterate over the chunks and generate summary for each chunk recursively
    for chunk in chunks:
        # Combine the previous context with the current chunk
        text = chunk
        
        # Tokenize the text
        inputs = tokenizer.encode(text, return_tensors='pt')
        
        # Generate summary using beam search
        outputs = model.generate(inputs, max_length=max_length, min_length=min_length,num_beams=num_beams, repetition_penalty=repetition_penalty, length_penalty=length_penalty, early_stopping=True, no_repeat_ngram_size=6)
        
        # Decode the summary
        summary_chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Add the summary chunk to the final summary
        summaryList.append(summary_chunk)
        summary += summary_chunk
        
        # Update previous context with the current chunk
        # prev_context = chunk
    
    return summaryList