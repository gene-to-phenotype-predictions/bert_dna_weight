
# Sequences to Weight Prediction
This project contain two different model to predict the change in weight a mouse will have from "knocking" out a single gene. The first model is a transformer model that uses the DNA Sequence to predict the weight change, and the second model uses the protien sequence. 
 
## Downloading project
Clone the project

```bash
  git clone https://github.com/gene-to-phenotype-predictions/bert_dna_weight.git
```

Go to the project directory

```bash
  cd bert_dna_weight
```

Install dependencies

```bash
  pip install requirements.txt
```

Finally setup your own wandb account that will log the details of your execution. You can follow instructions here: https://docs.wandb.ai/quickstart 


## Data 
All data can be accessed [here](https://drive.google.com/drive/folders/1Exv-jo6RlcHdD5fPYqSA0v3TN0FGducF?usp=share_link) or by following the download instructions below from the original sources

## DNA Sequence to Weight

For this model you need the following data files to be place into ./data/ folder

```bash
gene_symbol_dna_sequence.pkl
capstone_body_weight_Statistical_effect_size_analysis_genotype_early_adult_scaled_13022023_gene_symbol_harmonized.pkl
/data/gene_symbol_dna_sequence_exon.pkl

```

Then Open up the DNASeq2Weight.ipynb notebook, make sure your using the same venv you ran the requirements.txt file in. 

Replace the "entity" value in the file with your account name that you setup for wandb. For instance if account was "test-user" then it should look like: 

```python

wandb.init(project="DNA-Weight", entity="test-user",tags = ["custom_head"], config = config)

```
Finally you just need to "run all" in your notebook and the results will be logged to your personal wandb account. 

Note: wandb may prompt you to login the first time it encounters the "init" method, you should only have to do this once as it will save your key locally. 


## Protien Sequence to Weight
This model actually takes the contact map version of the protien sequences and uses that as the input. So in order to use this we must first convert our protien sequences into contact maps. 

You will need: 
1. Download all the data from https://www.uniprot.org/proteomes/UP000002494 and place them in the folder "MouseGenome"
2. Download and follow the setup instructions for https://github.com/kianho/pconpy Note: If your on windows use WSL and execute in the linux shell. 

Now we need to convert the .pbd files we got from Alpha Fold and convert them to contact maps. 

Navigating to the pconpy repo you just downloaded we can run the bash script: 

```bash
#!/bin/bash

# Change directory to the folder containing the .pdb.gz files
# Create the output folder if it doesn't exist
mkdir -p ./m_images

file_count=$(ls ../MouseGenome/ | wc -l)

# Loop through every file with the extension .pdb.gz and unzip them into the ../MousePDB folder
i=0
for file in ../MouseGenome/*.pdb
do
    i=$((i+1))
    echo "Processing file $i of $file_count: $file"
    python3 ./pconpy/pconpy.py dmap --pdb "$file" --output ./m_images/"$file".png --measure minvdw --no-colorbar --transparent --width-inches 3.88 --height-inches 3.9
done
```

To automatically convert the .pbd files into images.

Save the images into the ./data/MouseImages/ folder. 

Now we are ready to execute the model. Just like in the previous model make sure you have setup your wandb account and change the entity to your account name. 

Open Protien.ipynb file and run all, the results should be automatically logged to your wandb account. 



# HyperParameter Tuning 
In each file there is either a config, or sweep config dictionary. To adjust any of the hyperparemeters, edit this dictionary. 

Also in the Protien.ipynb notebook at the end there are 3 method calls 

```python
run_k_fold(init_config = config)
wandb.agent(sweep_id, train_loop, count=5)
train_loop(config)
```
run_k_fold will use the config dictionary and do a 6-fold analysis. 

wandb.agent will start a single thread for the wandb sweep. To run multiple threads on different machines you can run the same code with the same sweep_id to speed up execution time

train_loop(config) will just run a single training loop with the hyperparemeters defined in the config dictionary. 





# Results 
[Protien Model](https://wandb.ai/pcoady/Protein-Weightt/sweeps)

[DNA Model](https://wandb.ai/pcoady/DNA-Weight?workspace=default)











