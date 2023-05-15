
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