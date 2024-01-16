# BERT_Minimiser
MOTIVATION: To alter sentences used in speech decoding tasks for brain computer interfaces. The assumption is this different BERT scores -> different semantic meanings -> different brain regions activite. We want to design a set of sentences which produce very different invoked responses in the brain. 

Takes sentences and calculates the BERT score between them, it then modifies them until they have a minimised BERT score so they are semantically different. The model 'roberta-large' is used to calculate BERT. The code requires the number of desired output sentences, the input data will be reduced down to this number. 

Include a folder named 'data' with a text file that contains the sentences you want to find maximise the difference between. 
The sentences with the largest semantic difference/BERT score will be placed in the 'processed_data' folder.  

# Installation
1. Install the required packages: 
pip install -r requirements.txt


