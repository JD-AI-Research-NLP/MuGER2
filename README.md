# Main Requirements:
pytorch==1.5.0

transformers==2.6.0

# Paper result:
You need to download the [WIKI-LINKS](https://github.com/wenhuchen/WikiTables-WithLinks), and put it in **./processed_data/**.

Unzip the WIKI-LINKS and the retrieval_score_large.zip.

Download the [trained MRC models](https://pan.baidu.com/s/1bgUEMEBCi_V-sH7Yd1C6Og) to folder **./trained_model/**. <提取码：tftr>

Download the pre-trained [BERT-large](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin) model to folder **./muger2/bertlarge/**.

After preparing the trained models, the paper results can be obtained through the following commands.

    cd muger2
    python e-selector.py --mode dev --mrc_mode large --input_path ../processed_data/retrieval_scores_large.json --output_path predictions_large.json
    python evaluate_script.py predictions_large.json dev_reference.json
