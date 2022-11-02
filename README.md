# Update:
The code of the E-selector has been released.
Before we release the code of the Multi-G Retriever, we directly provide the retrieval_scores_large.json for E-sel reasoning.

# Main Requirements:
pytorch==1.5.0
transformers==2.6.0

# Paper result:
Download the trained MRC models to folder ./trained_model/
Download pre-trained BERT-large(base) model to folder ./bertlarge(base)/
After preparing the trained models, the paper results can be obtained through the following commands

    python e-selector.py --mode dev --mrc_mode large --input_path ../processed_data/retrieval_scores_large.json --output_path predictions_large.json
    python evaluate_script.py predictions_large.json dev_reference.json
