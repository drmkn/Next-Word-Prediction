## Next Word Prediction

In this project, I have used a linearly interpolated trigram model (with equal weights for unigram, bigram, and trigram) for next word prediction. Additionally, I have trained LSTM and BiLSTM models for the same task.

I have curated datasets from Kaggle for the corpus:
- [Dataset1: IELTS Writing Scored Essays](https://www.kaggle.com/datasets/mazlumi/ielts-writing-scored-essays-dataset)
- [Dataset2: NIPS 2015 Papers](https://www.kaggle.com/datasets/benhamner/nips-2015-papers?select=Papers.csv)

### How to Run

To generate next word predictions using the `generate_sentence.py` script, run the script and follow the prompts to select the model and input text:

```bash
python generate_sentence.py
```

### Sample Runs

Take `n` as the number of next words to be generated autoregressively.

1. **Input text**: `"life is"`, **n** = 5
   - **Trigram output**: `"life is rough you are not so"`
   - **LSTM output**: `"life is rough you will learn their"`
   - **BiLSTM output**: `"life is rough you are the water"`

2. **Input text**: `"gradient descent algorithm"`, **n** = 7
   - **Trigram output**: `"gradient descent algorithm for the first year which is the"`
   - **LSTM output**: `"gradient descent algorithm for logistic methods we show concentrate on"`
   - **BiLSTM output**: `"gradient descent algorithm provably requires solving a sparse recovery minimization"`
   

