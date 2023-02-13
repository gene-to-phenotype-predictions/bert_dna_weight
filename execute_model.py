from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModel, TrainingArguments, Trainer
from load_data import create_dataset
import evaluate
import argparse


mse_metric = evaluate.load("mse")


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
    model = load_model()
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

