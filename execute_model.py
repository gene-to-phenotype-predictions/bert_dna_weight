from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModel, TrainingArguments, Trainer
#from load_data import create_dataset
import evaluate
import argparse
import pandas as pd
from datasets import Dataset
from google.cloud import storage


mse_metric = evaluate.load("mse")
tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNA_bert_6')


def get_args():
  '''Parses args.'''

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--epochs',
      required=False,
      default=3,
      type=int,
      help='number of epochs')
  parser.add_argument(
      '--job_dir',
      required=True,
      type=str,
      help='bucket to store saved model, include gs://')
  args = parser.parse_args()
  return args


def reduce_data(df):
    df = df[["dna_seq","est_f_ea"]]
    df = df.rename({"est_f_ea":"label"}, axis=1)
    return Dataset.from_pandas(df)

def tokenize_function(df):
    return tokenizer(df["dna_seq"], padding=True, truncation=True, max_length=512)#512

def create_dataset():
    


    #client = storage.Client(
    #bucket = client.get_bucket('huggingface-pat-bucket')
    #train = pd.read_csv(bucket.get_blob('cleaned/train.csv')).sample(frac=1)
    train = pd.read_csv("gs://huggingface-pat-bucket/cleaned/train.csv").sample(frac=1)
    val = pd.read_csv("gs://huggingface-pat-bucket/cleaned/val.csv").sample(frac=1)
    test = pd.read_csv("gs://huggingface-pat-bucket/cleaned/test.csv").sample(frac=1)
    
    train = reduce_data(train).map(tokenize_function, batched=True)
    val = reduce_data(val).map(tokenize_function, batched=True)
    test = reduce_data(test).map(tokenize_function, batched=True)
    
    return train,val,test


def load_model(device = "cpu"):
    model = AutoModelForSequenceClassification.from_pretrained('zhihan1996/DNA_bert_6', 
                                                           num_labels=1, 
                                                           ignore_mismatched_sizes=True).to(device)
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    #labels = labels.reshape(-1, 1)
    mse = mse_metric.compute(predictions=logits, references=labels)
    return mse

def main():
    args = get_args()
    training_args = TrainingArguments(output_dir=f'{args.job_dir}/model_output', 
                                  evaluation_strategy='epoch',
                                  per_device_train_batch_size = 5,  
                                  num_train_epochs=2)
    train, val, test = create_dataset()
    model = create_dataset()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model()
    
if __name__ == "__main__":
    main()

