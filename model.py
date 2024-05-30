# Import all the necessary classes and initialize the tokenizer and model.
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from tqdm import tqdm
import sys
import pdb
tokenizer = AutoTokenizer.from_pretrained("/home/cse/dual/cs5190446/.cache/huggingface/hub/models--ai4bharat--IndicNER/snapshots/b56cb5e9bdf37cb0d653e658b3d4dbe563bebf34/", max_length=2048)

model = AutoModelForTokenClassification.from_pretrained("/home/cse/dual/cs5190446/.cache/huggingface/hub/models--ai4bharat--IndicNER/snapshots/b56cb5e9bdf37cb0d653e658b3d4dbe563bebf34/", max_length=2048).cuda()

def put_on_gpu(data):
    for k in data:
        data[k] = data[k].cuda()
    return data


def get_predictions( sentence, tokenizer, model ):
  # Let us first tokenize the sentence - split words into subwords
  tok_sentence = tokenizer(sentence, return_tensors='pt')
  tok_sentence = put_on_gpu(tok_sentence)
  if(tok_sentence['input_ids'].shape[1] > 512):
    return None

  with torch.no_grad():
    # we will send the tokenized sentence to the model to get predictions
    logits = model(**tok_sentence).logits.argmax(-1)
    
    # We will map the maximum predicted class id with the class label
    predicted_tokens_classes = [model.config.id2label[t.item()] for t in logits[0]]
    
    predicted_labels = []
    
    previous_token_id = 0
    # we need to assign the named entity label to the head word and not the following sub-words
    word_ids = tok_sentence.word_ids()
    for word_index in range(len(word_ids)):
        if word_ids[word_index] == None:
            previous_token_id = word_ids[word_index]
        elif word_ids[word_index] == previous_token_id:
            previous_token_id = word_ids[word_index]
        else:
            predicted_labels.append( predicted_tokens_classes[ word_index ] )
            previous_token_id = word_ids[word_index]
    
    return predicted_labels

def predict(sentence, file):

    length = len(sentence.split(' '))
    # print("len", length)
    predicted_labels = get_predictions(sentence=sentence, 
                                   tokenizer=tokenizer,
                                   model=model
                                   )
    if(predicted_labels is None):
        return 
    x=len(sentence.split(' '))
    y=len(predicted_labels)
    # print(y)
    # if (x>y):
    #     exit()
    for index in range(min(x,y)):
        file.write( sentence.split(' ')[index] + '\t' + predicted_labels[index] )
        file.write("\n")

        # print(sentence.split(' ')[index] + '\t' + predicted_labels[index])
    file.write('\n')
    # print()

# sentence = 'लगातार हमलावर हो रहे शिवपाल और राजभर को सपा की दो टूक, चिट्ठी जारी कर कहा- जहां जाना चाहें जा सकते हैं'
with open(sys.argv[1], 'r') as file:
    # Read the first line
    lines = file.readlines()
    f = open(sys.argv[2], 'w')
    # Loop through each line until the end of the file
    for line in tqdm(lines):
        # Print the current line
        # print(line)
        predict(line.strip(), f)
        # Read the next line
        line = file.readline()
    f.close()

# predict(sentence)
