
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
model_name = 'bert-base-uncased'

# Load the pre-trained adapters we want to fuse
# model.load_adapter("nli/multinli@ukp", with_head=False)
# model.load_adapter("sts/qqp@ukp", with_head=False)
# model.load_adapter("sentiment/sst-2@ukp", with_head=False)
# model.load_adapter("comsense/winogrande@ukp", with_head=False)
# model.load_adapter("sentiment/imdb@ukp", with_head=False)
# model.load_adapter("comsense/hellaswag@ukp", with_head=False)

# model.load_adapter("comsense/siqa@ukp", with_head=False)
# model.load_adapter("comsense/cosmosqa@ukp", with_head=False)
# model.load_adapter("nli/scitail@ukp", with_head=False)
# model.load_adapter("argument/ukpsent@ukp", with_head=False)

# model.load_adapter("comsense/csqa@ukp", with_head=False)
# model.load_adapter("qa/boolq@ukp", with_head=False)
# model.load_adapter("sts/mrpc@ukp", with_head=False)

# model.load_adapter("nli/sick@ukp", with_head=False)
# model.load_adapter("nli/rte@ukp", with_head=False)
# model.load_adapter("nli/cb@ukp", with_head=False)

model.load_adapter("nli/multinli@ukp", load_as="multinli", with_head=False,model_name=model_name)
model.load_adapter("sts/qqp@ukp",  load_as="qqp", with_head=False, model_name=model_name,)
model.load_adapter("sentiment/sst-2@ukp",  load_as="sst-2", with_head=False, model_name=model_name)
model.load_adapter("comsense/winogrande@ukp",  load_as="winogrande", with_head=False, model_name=model_name)
model.load_adapter("sentiment/imdb@ukp",  load_as="imdb", with_head=False, model_name=model_name)
model.load_adapter("comsense/hellaswag@ukp", load_as="hellaswag", with_head=False, model_name=model_name)

model.load_adapter("comsense/siqa@ukp", load_as="siqa", with_head=False, model_name=model_name)
model.load_adapter("comsense/cosmosqa@ukp", load_as="cosmosqa",with_head=False, model_name=model_name)
model.load_adapter("nli/scitail@ukp", load_as="scitail",with_head=False, model_name=model_name)
model.load_adapter("argument/ukpsent@ukp", load_as="ukpsent",with_head=False, model_name=model_name)

model.load_adapter("comsense/csqa@ukp", load_as="csqa",with_head=False, model_name=model_name)
model.load_adapter("qa/boolq@ukp", load_as="boolq",with_head=False, model_name=model_name)
model.load_adapter("sts/mrpc@ukp", load_as="mrpc",with_head=False, model_name=model_name)

model.load_adapter("nli/sick@ukp", load_as="sick",with_head=False, model_name=model_name)
model.load_adapter("nli/rte@ukp", load_as="rte",with_head=False, model_name=model_name)
model.load_adapter("nli/cb@ukp", load_as="cb",with_head=False, model_name=model_name)


model.save_all_adapters("./model/model_adapters2")
