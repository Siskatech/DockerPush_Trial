
import random
import pandas as pd
import nltk
import pyodbc
import warnings
import re
warnings.filterwarnings('ignore')
from flask import Flask, request, jsonify, Response

from libraries import utils_preprocess as upp, utils_ngram as ung, utils_perplexity as up, utils_suggestions as suggest

conx_string = "DRIVER={ODBC Driver 17 for SQL Server}; SERVER=APCKRMPTMD01PV,41433; Database=Namlos; UID=Namlos_user; PWD=M@!!eL@498#;"
conx = pyodbc.connect(conx_string)
query = '''SELECT [Issue]
  FROM [Namlos].[dbo].[KB_View_Issue_PowerBI]'''


df = pd.read_sql(query, conx)
df.dropna(subset=['Issue'], inplace=True)

def text_clean(text):
    # Remove whitespace
    text = text.strip()
      
    # Remove special characters and punctuation
    text = re.sub(r'[^A-Za-z\s]','', text)
    text = re.sub(r'\b(\w+)\.{3}', r'\1', text)

   
    # Convert to lowercase
    text = text.lower()

    # Remove Number 
    text = re.sub('[0-9]+', '', text)

    # Replace Word 
    text = text.replace('5 S', '5s')
    text = text.replace('5S', '5s')
    text = text.replace('improer','improper')


    # Remove 2 of word repetitif
    pattern = r'\b(\w+)(2)\b'
    text = re.sub(pattern, lambda match: f'{match.group(1)}', text)
   
    return text

df['Issue'] = df['Issue'].apply(text_clean)

data = ' '.join(df['Issue'])

tokenized_data = upp.get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]

train_data, test_data, vocab = upp.preprocess_data(train_data, test_data, count_threshold=2)

n_gram_counts_list = []
for n in range(1, 6):
    n_model_counts = ung.count_n_grams(train_data, n)
    n_gram_counts_list.append(n_model_counts)

app = Flask(__name__)
@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    data = request.json
    previous_tokens = data.get('previous_tokens', [])

    suggestions = suggest.get_suggestions(previous_tokens, n_gram_counts_list, vocab, k=1.0)
    unique_words = {suggestion[0] for suggestion in suggestions}
    unique_suggest_words = [[word] for word in unique_words]
    
    return jsonify(unique_suggest_words)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)