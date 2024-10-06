import json
from training_models.lstm_bilstm import LSTM_Model
import torch
import re
import pickle
from nltk import word_tokenize

a, b, c = 1/3,1/3,1/3 #scaling factors for the trigram model

with open('./training_models/n_grams_probabilities.pkl', 'rb') as f:
    probs = pickle.load(f)
P_1, P_2, P_3 = probs[0], probs[1], probs[2]

with open("./training_models/word_freq.json", "r") as json_file:
    word_freq = json.load(json_file)
with open("./training_models/word_to_idx.json", "r") as json_file:
    word_to_idx = json.load(json_file)
with open("./training_models/idx_to_word.json", "r") as json_file:
    idx_to_word = json.load(json_file)    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize((re.sub(r'([.,!?@#:$%^&*()_+=-])',' ',text)))
    return words

#Autoregressively complete a sentence given atleast two words for trigram model
def complete_sentence_trigram(text,n,show_top_5=False):
    for i in range(n):
        if i == 0:
            words = preprocess_text(text)
        w1,w2 = words[-2], words[-1] #context words
        pred = dict()
        for w in P_3.keys():
            if w1 == w[0] and w2 == w[1]:
                w3 = w[2]
                pred[(w1,w2,w3)] = c*P_1[(w1,)] + b*P_2[(w1,w2)] + a*P_3[(w1,w2,w3)]           
        pred_5 = sorted(pred.items(), key = lambda x : -pred[x[0]])[:5]
        if not len(pred_5):
            print("The last 2 words don't exist as bigram in the corpus not able to generate further. Exiting...") 
            return
        top_5 = [pred[0][2] for pred in pred_5]
        if show_top_5:
            print(f"top five predicted words --> {top_5}")
        best_pred = top_5[0]
        words.append(best_pred)
        text = text + ' ' + best_pred
        print(f"updated sentence --> {text}\n")
    return 


#for LSTM and BiLSTM
def predict_next_words(model, input_text,n,show_top_5=False,idx_to_word = idx_to_word, word_to_idx = word_to_idx): 
    text = input_text.lower()
    model.eval()
    with torch.no_grad():
        for i in range(n):
            if i == 0:
                tokens = preprocess_text(text)
                input_ids = [word_to_idx[token] for token in tokens]
            else:
                input_ids.append(best_prediction.item())  
            # print(input_ids)      
            outputs = model(torch.tensor(input_ids).unsqueeze(0).to(device))
            predictions = torch.topk(outputs, dim=1,k=5)[1]
            # print(predictions[0])
            best_prediction = predictions[0][0]        
            predicted_words = [idx_to_word.get(str(idx.item()), '<UNK>') for idx in predictions[0]]
            if show_top_5:
                print(f"top five predicted words - {predicted_words}")
            text = text + ' ' + predicted_words[0] 
            print(f"updated sentence --> {text}\n")   
  
    return text

def generate_sentence(text,model,n,show_top_5 = False,idx_to_word = idx_to_word, word_to_idx = word_to_idx):
    vocab_size  = len(word_to_idx)

    embedding_dim = 256
    hidden_size = 128
    num_layers = 2          
    lstm_dropout = 0.3
    if model == 'lstm':
        lstm_model = LSTM_Model(
                    embedding_dim=embedding_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    vocab_size=vocab_size,
                    lstm_dropout=lstm_dropout,
                    bidirectional=False
                     )
        lstm_model.to(device)
        lstm_check_point_path = './training_models/model_checkpoints/lstm/best_model.pth'
        lstm_check_point = torch.load(lstm_check_point_path,weights_only = True)
        lstm_model.load_state_dict(lstm_check_point['model_state_dict'])
        predict_next_words(model = lstm_model,input_text = text,n = n, show_top_5 = show_top_5,idx_to_word = idx_to_word, word_to_idx = word_to_idx)

    elif model == 'bilstm':
        bilstm_model = LSTM_Model(
                    embedding_dim=embedding_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    vocab_size=vocab_size,
                    lstm_dropout=lstm_dropout,
                    bidirectional=True
                     )
        bilstm_model.to(device)
        bilstm_check_point_path = './training_models/model_checkpoints/bilstm/best_model.pth'
        bilstm_check_point = torch.load(bilstm_check_point_path,weights_only = True)
        bilstm_model.load_state_dict(bilstm_check_point['model_state_dict'])
        predict_next_words(model = bilstm_model,input_text = text,n = n, show_top_5 = show_top_5,idx_to_word = idx_to_word, word_to_idx = word_to_idx)

    elif model == 'trigram':
        complete_sentence_trigram(text,n,show_top_5=show_top_5)

    return



if __name__ == "__main__":
    select_model = int(input("Type 1 for trigram, 2 for lstm or 3 for bilstm\n"))
    text = input("Type atleast 2 words\n")
    n = int(input("Type number of words to generate autoregressively\n"))
    models = {1:"trigram", 2:"lstm",3:"bilstm"}
    generate_sentence(model=models[select_model],text=text,n=n,show_top_5 = True)
