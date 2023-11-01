
from datasets import load_dataset
dataset = load_dataset('imdb')
dataset.save_to_disk('./data')

