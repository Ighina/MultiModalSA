# MultiModalSA
MultiModal Sentiment Analysis architectures for CMU-MOSEI.

# Description

The repository contains four multimodal architectures and relative training and test functions for sentiment analysis on CMU-MOSEI. Inside the data folder, transcriptions and labels are provided for the standard training, validation and test sentences as from the [original repository](https://github.com/A2Zadeh/CMU-MultimodalSDK/tree/master/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI). BERT embeddings (text modality), COVAREP features (audio modality) and FACET features (video modality) can be downloaded via the following links:

- BERT embeddings: https://drive.google.com/file/d/13y2xoO1YlDrJ4Be2X6kjtMzfRBs7tBRg/view?usp=sharing
- COVAREP: https://drive.google.com/file/d/1XpRN8xoEMKxubBHaNyEivgRbnVY2iazu/view?usp=sharing
- FACET: https://drive.google.com/file/d/1BSjMfKm7FQM8n3HHG5Gn9-dTifULC_Ws/view?usp=sharing

After downloading, move the files to the data folder, before using the programme.

To extract your own features, follow the procedure as described in the [original repository](https://github.com/A2Zadeh/CMU-MultimodalSDK). The programme as it is now requires the extracted features to be in a specific format (see later the examples for how to change the Hyperparameters.json file): if a different format is needed change the relative lines in the Train.py and Test.py functions.

# Usage

The programme allows for the creation of separate experiment folders, so that different hyperparameters/models can be tested. After having downloaded/extracted the features and having installed all the required libraries, an experiment folder can be created with the following line in the terminal:
```
Prepare_workspace.py -name {new_experiment_folder_name}
```
The above line calls the Prepare_workspace.py function, that creates the experiment folder containing the standard folders for storing results and saved models, as well as the Hyperparameters.json file that can be changed in order to choose architecture/hyperparameters of the model.
After having created the experiment folder, the model as specified in the Hyperparameters.json file in the relative folder can be trained using:
```
run.py -folder {new_experiment_folder_name}
```
When training is complete and if the save_model option is set to TRUE in the Hyoerparameters.json file, then the trained model can be tested on test data with the following line:
```
run_test.py -folder {new_experiment_folder_name}
```
The function prints the results on the terminal, but the option save_output in the Hyperparameters.json file can be set to TRUE, as well, so that the results for all the metrics are saved in the output.csv file in the output folder inside the experiment one.

# Hyperparameters
The Hyperparameters.json file is copied into each new experiment folder when the Prepare_workspace.py function is called and it can be changed in order to experiment different parameters/architectures. The master file (i.e. the one that is copied in the new experiment folder) already contains all the options and default values for them, acting as a template. If the options are not changed, the run.py function will train the attention-based model with the pre-specified hyperparameters. By running the following line in the terminal, a description of all available options will be printed out:
```
Train.py -h
```
Some example of how the Hyperparameters.json file can be modified to achieve different results is included below:

- Use early fusion architecture instead of attention-based (same for all other architectures, with relative option being changed to "TRUE"): Change "--AttentionModel": "TRUE" to "--AttentionModel": "FALSE" and "--EarlyFusion": "FALSE" to "EarlyFusion": "TRUE"

- Change audio features file (same for video features file): Change "--audio_file": "COAVAREP_aligned_MOSEI.pkl" to "--audio_file": "{new_audio_file_name}.pkl" (the new audio file must be a pkl file containing a dictionary with the opinion segments' ids as keys and the relative features as values)

- Change text features file: Change "--text_file": "BERT_MOSEI.pkl" to "--text_file": "{new_text_file_name}.pkl" (currently, the text file needs to be a dictionary having "data" as key and, as the relative value, a list with all the sentence-level features in sequential order, according to how they appear in the training, validation and test .tsv files)
