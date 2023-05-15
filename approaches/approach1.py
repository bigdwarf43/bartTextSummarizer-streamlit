
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
