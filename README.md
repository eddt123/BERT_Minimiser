# BERT_Minimiser
MOTIVATION: To alter sentences used in speech decoding tasks for brain computer interfaces. The assumption is this different BERT scores -> different semantic meanings -> different brain regions activite. We want to design a set of sentences which produce very different invoked responses in the brain. 

Takes sentences and calculates the BERT score between them, it then modifies them until they have a minimised BERT score so they are semantically different. 
Masking is used to regenerate the sentences repeatedly until a minimal BERT score is found. The model 'roberta-large' is used to calculate BERT and for the masking. 

Include a folder named 'data' with a list of text files that contain the sentences you want to maximise the difference between. 
The modified sentences will be placed in the 'processed_data' folder.  

# Requirements
pip install -r requirements.txt
