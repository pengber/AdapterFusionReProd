# -*- coding: utf-8 -*-

import datasets
from datasets import load_dataset

# dataset = load_dataset("super_glue", "cb")
dataset = load_dataset("./data/super_glue/cb")

print(dataset.num_rows)

"""Every dataset sample has a premise, a hypothesis and a three-class class label:"""

dataset['train'].features

"""Now, we need to encode all dataset samples to valid inputs for our `bert-base-uncased` model. Using `dataset.map()`, we can pass the full dataset through the tokenizer in batches:"""

from transformers import BertTokenizer
model_path = './model/bert-base-uncased/'
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_path)

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(
      batch["premise"],
      batch["hypothesis"],
      max_length=180,
      truncation=True,
      padding="max_length"
  )

# Encode the input data
dataset = dataset.map(encode_batch, batched=True)
# The transformers model expects the target class column to be named "labels"
dataset = dataset.rename_column("label", "labels")
# Transform to pytorch tensors and only output the required columns
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

"""New we're ready to setup AdapterFusion...

## Fusion Training

We use a pre-trained BERT model from HuggingFace and instantiate our model using `BertModelWithHeads`.
"""

from transformers import BertConfig, BertModelWithHeads

id2label = {id: label for (id, label) in enumerate(dataset["train"].features["labels"].names)}

config = BertConfig.from_pretrained(
    model_path,
    id2label=id2label,
)
model = BertModelWithHeads.from_pretrained(
    model_path,
    config=config,
)
"""Now we have everything set up to load our _AdapterFusion_ setup. First, we load three adapters pre-trained on different tasks from the Hub: MultiNLI, QQP and QNLI. As we don't need their prediction heads, we pass `with_head=False, model_name=model_name` to the loading method. Next, we add a new fusion layer that combines all the adapters we've just loaded. Finally, we add a new classification head for our target task on top."""

from transformers.adapters.composition import Fuse
model_adapter_dir = 'model/model_adapters/'
# Load the pre-trained adapters we want to fuse
# lang_adapter_config = AdapterConfig.load("pfeiffer", non_linearity="gelu", reduction_factor=2)
model.load_adapter(model_adapter_dir +"multinli", load_as="multinli", with_head=False,model_name=model_name)
model.load_adapter(model_adapter_dir +"qqp",  load_as="qqp", with_head=False, model_name=model_name,)
model.load_adapter(model_adapter_dir +"sst-2",  load_as="sst-2", with_head=False, model_name=model_name)
model.load_adapter(model_adapter_dir +"winogrande",  load_as="winogrande", with_head=False, model_name=model_name)
model.load_adapter(model_adapter_dir +"imdb",  load_as="imdb", with_head=False, model_name=model_name)
model.load_adapter(model_adapter_dir +"hellaswag", load_as="hellaswag", with_head=False, model_name=model_name)

model.load_adapter(model_adapter_dir +"siqa", load_as="siqa", with_head=False, model_name=model_name)
model.load_adapter(model_adapter_dir +"cosmosqa", load_as="cosmosqa",with_head=False, model_name=model_name)
model.load_adapter(model_adapter_dir +"scitail", load_as="scitail",with_head=False, model_name=model_name)
model.load_adapter(model_adapter_dir +"ukpsent", load_as="ukpsent",with_head=False, model_name=model_name)

model.load_adapter(model_adapter_dir +"csqa", load_as="csqa",with_head=False, model_name=model_name)
model.load_adapter(model_adapter_dir +"boolq", load_as="boolq",with_head=False, model_name=model_name)
model.load_adapter(model_adapter_dir +"mrpc", load_as="mrpc",with_head=False, model_name=model_name)

model.load_adapter(model_adapter_dir +"sick", load_as="sick",with_head=False, model_name=model_name)
model.load_adapter(model_adapter_dir +"rte", load_as="rte",with_head=False, model_name=model_name)
model.load_adapter(model_adapter_dir +"cb", load_as="cb",with_head=False, model_name=model_name)

#multinli,qqp,sst-2,winogrande,imdb,hellaswag,siqa,cosmosqa,scitail,ukpsent,csqa,boolq,mrpc,sick,rte,cb

# Add a fusion layer for all loaded adapters
# Fuse("multinli", "qqp", "sst-2", "winogrande", "imdb", "hellaswag", "siqa", "cosmosqa", "scitail", "ukpsent", "csqa", "boolq", "mrpc", "sick", "rte", "cb")
# model.add_adapter_fusion(Fuse("multinli", "qqp", "qnli"))
model.add_adapter_fusion(Fuse("multinli", "qqp", "sst-2", "winogrande", "imdb", "hellaswag", "siqa", "cosmosqa", "scitail", "ukpsent", "csqa", "boolq", "mrpc", "sick", "rte", "cb"))
# model.set_active_adapters(Fuse("multinli", "qqp", "qnli"))
model.set_active_adapters(Fuse("multinli", "qqp", "sst-2", "winogrande", "imdb", "hellaswag", "siqa", "cosmosqa", "scitail", "ukpsent", "csqa", "boolq", "mrpc", "sick", "rte", "cb"))

# Add a classification head for our target task
print("len of id2lable", len(id2label))
model.add_classification_head("cb", num_labels=len(id2label))

"""The last preparation step is to define and activate our adapter setup. Similar to `train_adapter()`, `train_adapter_fusion()` does two things: It freezes all weights of the model (including adapters!) except for the fusion layer and classification head. It also activates the given adapter setup to be used in very forward pass.

The syntax for the adapter setup (which is also applied to other methods such as `set_active_adapters()`) works as follows:

- a single string is interpreted as a single adapter
- a list of strings is interpreted as a __stack__ of adapters
- a _nested_ list of strings is interpreted as a __fusion__ of adapters
"""

# Unfreeze and activate fusion setup
adapter_setup = Fuse("multinli", "qqp", "sst-2", "winogrande", "imdb", "hellaswag", "siqa", "cosmosqa", "scitail", "ukpsent", "csqa", "boolq", "mrpc", "sick", "rte", "cb")
model.train_adapter_fusion(adapter_setup)

"""For training, we make use of the `Trainer` class built-in into `transformers`. We configure the training process using a `TrainingArguments` object and define a method that will calculate the evaluation accuracy in the end. We pass both, together with the training and validation split of our dataset, to the trainer instance."""

import numpy as np
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction

training_args = TrainingArguments(
    learning_rate=5e-5,
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
    evaluation_strategy="epoch"
)

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}
trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_accuracy
)

"""Start the training 🚀 (this will take a while)"""

trainer.train()

"""After completed training, let's check how well our setup performs on the validation set of our target dataset:"""

# trainer.evaluate()

"""We can also use our setup to make some predictions (the example is from the test set of CB):"""
print("predict")
import torch

def predict(premise, hypothesis):
  encoded = tokenizer(premise, hypothesis, return_tensors="pt")
  if torch.cuda.is_available():
    encoded.to("cuda")
  logits = model(**encoded)[0]
  pred_class = torch.argmax(logits).item()
  return id2label[pred_class]

predict_result = predict("""
``It doesn't happen very often.'' Karen went home
happy at the end of the day. She didn't think that
the work was difficult.
""",
"the work was difficult"
)
print("predict result is {}",predict_result)
"""Finally, we can extract and save our fusion layer as well as all the adapters we used for training. Both can later be reloaded into the pre-trained model again."""

model.save_adapter_fusion("./saved", "multinli,qqp,sst-2,winogrande,imdb,hellaswag,siqa,cosmosqa,scitail,ukpsent,csqa,boolq,mrpc,sick,rte,cb")
model.save_all_adapters("./saved")

#!ls -l saved

"""That's it. Do check out [the paper on AdapterFusion](https://arxiv.org/pdf/2005.00247.pdf) for a more theoretical view on what we've just seen.

➡️ `adapter-transformers` also enables other composition methods beyond AdapterFusion. For example, check out [the next notebook in this series](https://colab.research.google.com/github/Adapter-Hub/adapter-transformers/blob/master/notebooks/04_Cross_Lingual_Transfer.ipynb) on cross-lingual transfer.
"""
