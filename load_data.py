import pandas as pd
import numpy as np
import pickle
import Bio   
from jproperties import Properties
from motif_utils import seq2kmer # Soruced from https://github.com/jerryji1993/DNABERT
from transformers import AutoTokenizer
from datasets import Dataset
import evaluate
import argparse

import json

tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNA_bert_6')

def explode_dna(df):
    df.dna_seq = df.dna_seq.str.split(" ")
    df.dna_seq = df.apply(lambda x: [" ".join(x["dna_seq"][i:i + 512]) for i in range(0, len(x["dna_seq"]), 512)], axis=1).copy()
    df = df.explode('dna_seq')
    return df


def load_data(all_data = False, explode = False, save = False):
    '''
    1. Load the DNA Seq
    2. Convert them to k-mers (Credit to jerryji1993)
    3. Read body weight
    4. Merge data together
    5. Remove: Zygosity, and p-value threshold?
    6. split each dna seq into 512 size chunks for ML model, give each chunk new row
    
    '''
    configs = Properties()
    with open("env.properties", "rb") as config_file:
        configs.load(config_file)
    
    with open(configs.get("SEQ_LOCATION").data, 'rb') as file:
        dna_dict =  pickle.load(file)
    with open(configs.get("BODY_WEIGHT_LOCATION").data, 'rb') as file:
        df =  pickle.load(file)
        
    dna = pd.Series({k:seq2kmer(str(v.seq), 6) for k,v in dna_dict.items()})
    dna.name = "dna_seq"
    df = df.merge(dna, left_on="gene_symbol", right_index=True)
    df.to_csv("cleaned/full_dataset.csv")

    #Randomize the order
    df = df.sample(frac=1)
    df_len = len(df)
    
    
    
    if all_data:
        train = df[:int(np.round(df_len*.8))].copy()
        val = df[int(np.round(df_len*.8)):int(np.round(df_len*.9))].copy()
        test = df[int(np.round(df_len*.9)):].copy()
    else:
        train = df[:int(np.round(df_len*.1))].copy()
        val = df[int(np.round(df_len*.1)):int(np.round(df_len*.15))].copy()
        test = df[int(np.round(df_len*.15)):int(np.round(df_len*.2))].copy()
        
        
        
    if explode:
        train = explode_dna(train)
        val = explode_dna(val)
        test = explode_dna(test)
        
    if save:
        train.to_csv("cleaned/train.csv")
        val.to_csv("cleaned/val.csv")
        test.to_csv("cleaned/test.csv")
    return train,val,test


def reduce_data(df):
    df = df[["dna_seq","est_f_ea"]]
    df = df.rename({"est_f_ea":"label"}, axis=1)
    return Dataset.from_pandas(df)

def tokenize_function(df):
    return tokenizer(df["dna_seq"], padding=True, truncation=True, max_length=512)#512

def create_dataset():
    train = pd.read_csv("cleaned/train.csv").sample(frac=.1)
    val = pd.read_csv("cleaned/val.csv").sample(frac=1)
    test = pd.read_csv("cleaned/test.csv").sample(frac=1)
    
    train = reduce_data(train).map(tokenize_function, batched=True)
    val = reduce_data(val).map(tokenize_function, batched=True)
    test = reduce_data(test).map(tokenize_function, batched=True)
    
    return train,val,test


def main():
    train, val,  test = create_dataset()


if __name__ == "__main__":
    main()
    




    

