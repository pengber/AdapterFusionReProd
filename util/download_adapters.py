
from datasets import load_dataset
# dataset = load_dataset("super_glue", "cb", cache_dir='../data')
# # dataset.save_to_disk('../data')
from transformers import BertConfig, BertModelWithHeads

# id2label = {id: label for (id, label) in enumerate(dataset["train"].features["labels"].names)}

id2label = {}
config = BertConfig.from_pretrained(
    "bert-base-uncased",
    id2label=id2label,
)
model = BertModelWithHeads.from_pretrained(
    "bert-base-uncased",
    config=config,
)


# Load the pre-trained adapters we want to fuse
model.load_adapter("nli/multinli@ukp", with_head=False)
model.load_adapter("sts/qqp@ukp", with_head=False)
model.load_adapter("sentiment/sst-2@ukp", with_head=False)
model.load_adapter("comsense/winogrande@ukp", with_head=False)
model.load_adapter("sentiment/imdb@ukp", with_head=False)
model.load_adapter("comsense/hellaswag@ukp", with_head=False)

model.load_adapter("comsense/siqa@ukp", with_head=False)
model.load_adapter("comsense/cosmosqa@ukp", with_head=False)
model.load_adapter("nli/scitail@ukp", with_head=False)
model.load_adapter("argument/ukpsent@ukp", with_head=False)

model.load_adapter("comsense/csqa@ukp", with_head=False)
model.load_adapter("qa/boolq@ukp", with_head=False)
model.load_adapter("sts/mrpc@ukp", with_head=False)

model.load_adapter("nli/sick@ukp", with_head=False)
model.load_adapter("nli/rte@ukp", with_head=False)
model.load_adapter("nli/cb@ukp", with_head=False)

model.save_all_adapters("./model_adapters")
