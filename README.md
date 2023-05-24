# bartTextSummarizer-streamlit

Add Model path to lsaApproach.py

terms_and_conditions_pipeline = pipeline(
                task='summarization',
                model="sshleifer/distilbart-xsum-12-3", //HERE
                tokenizer="sshleifer/distilbart-xsum-12-3" //HERE
            )

##To Run
streamlit run lsaApproach.py
