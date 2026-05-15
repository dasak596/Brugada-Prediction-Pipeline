# Brugada CNN Interpretability
This project generates interpretability reports for multiple fine-tuned BERT-like models using transformer-based and CNN-based classifiers.

## Description
Given a series of pretrained models and tokenizers, the finetune script (no improved for gpu usage) runs interpretability analyses and outputs LaTeX reports that summarize model attention and feature importance.

## Requirements
Full requirements list can be explored in detail in requirements.txt. I used a python virtual environment, that I called myenv, for incompatibility with cluster nodes compatability (Not all nodes had the same software installed, so I had to enable this virtual environment to run the scripts in the node available)



## Installation of myenv
python -m venv myenv

If you want to enter from a login or interactive node: source myenv/bin/activate

To obtain its specifications: pip install -r requirements.txt

Install libraries until the your requirements match my requirements.


I have all python programs located in a subdirectory of my working directory called "programs", where results also will be deployed. This can be changed, but then sbatch files must be adapted. 

## Cardiology Domain Fine-tuning
python finetune_bert_EnglishModelsFixed
For each bert-like model, we have to run the finetuning script to get the respective finetuned model. 
The model github repository (or local repository in case of previous download) is set using MODEL_NAME variable (Ex: MODEL_NAME = "./ClinicalBERT").
The resulting model will be deployed on a directory called "fine_tuned_{model_name}". It can be changed using SAVE_MODEL_BASE_PATH (Ex: SAVE_MODEL_BASE_PATH = "./fine_tuned_SpanishBERT"

The finetuned model itself will be used for transformer classification while the finetuned tokenizer will be used for CNN (automatically done).
I think that there were some incompabilities between gpu and some of the original bert-like models when I tried to finetune them at the start of the TFM, so I ended running this on my computer (without gpu) and then uploading the resulting finetuned models to the cluster.

## Transformer-based and CNN-based classifiers Fine-tuning
sbatch launch_all_models.sh

The launch_all_models.sh lists all used models and launch another sbatch file (run_model.sh) for each of the CNN and transformer models described in that list.
Then run_model.sh runs the python file associated with transformer classification or cnn classification depending on the case.
This is done in two different sbatch files instead of one alone to prevent the best model finetuning to be done for all models and architectures at same time, which results on a kill because of an OUT OF MEMORY (OOM)
With the actual 2-sbatch files configuration, the gpu only runs in parallel the number of models that can be encapsulated in their RAM (If GPU has 128GB available and we use partitions of 64GB as specified, it will run 2 at a time)
This allows to parallelize the computations while avoid OOM problems (and instead of python gpu parallelization, this allows to parallelize in the same gpu if it has enough space).
I think partition space can be downgraded even to 32GB and biggest model (roberta-large) will still not fall in OOM kill, but I'm not sure.


For Transformers-based classification, before running the model, it is needed to create the respective directory, called the same way as in the launch_all_models.sh file (Ex: The directory name will be classificationLayer_based_model_SpanishClinicalROBERT_512 for SpanishClinicalROBERT model, which is also the name of the directory specified in launch_all_models.sh). The "_512" addition after model name was made by me in order to differentiate it from the previous models executed using 128 tokens, it can be removed from sbatches and directores, but it is necessarily then to remove from all of them, as they are interconnected.
Inside the classificationLayer_based_model_SpanishClinicalROBERT_512, the directory "original_model" must be created and we must locate inside it the respective cardiology domain finetuned resulting model.
The best model obtained by the classificatior finetuning script will also be stored inside.

For CNN-based classification, the same directory has to be created (ex: cnn_based_model_finetunedSpanishClinicalROBERT_512 for SpanishClinicalROBERT model). But in this one we do not store finetuned transformer model or tokenizer model, we let it empty for storing best model (no subdirectory best_model has to be created here)
The sbatch is already told to use the finetuned tokenizer from the finetuned cardiology domain directory related to the model used (Ex: for SpanishClinicalROBERT model it will use fine_tuned_SpanishClinicalROBERT).


## Transformers XAI
sbatch run_TransformerSHAPexplanations.sh
It will read input from classificationLayer_based_model_{model_name}_512/best_model.
Results will consist on text and plots and will be deployed on classificationLayer_based_model_{model_name}_512_plots (Ex: classificationLayer_based_model_SpanishClinicalROBERT_512_plots)
Text results will consisit on the top 10 sentences for TP, TN and FN cases and their respectives scores.
For each of those cases, each top 10 sentences plots will be stored in its respective TP, TN or FN subdirectory (automatically created)

## CNN XAI
sbatch run_CNNexplanations.sh
It will read input from cnn_based_model_finetuned{model_name}_512/best_model.
Results will consist on a tex file called interpretability_report_cnn_based_model_finetuned{model_name}.tex (Ex: for SpanishClinicalROBERT model it will be interpretability_report_cnn_based_model_finetunedSpanishClinicalROBERT_512) listing each filter size top 10 sentences for TP, TN, and FN cases.

