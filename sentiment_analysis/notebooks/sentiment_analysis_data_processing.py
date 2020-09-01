
# load the path
import sys
sys.path.append('/Users/poudel/opt/miniconda3/envs/nlp/lib/python3.7/site-packages')

# load the libraries
import numpy as np
import pandas as pd
import time
import re
import string
from urllib.parse import urlparse
import multiprocessing as mp
import nltk
from nltk.corpus import stopwords

time_start = time.time()

# Load the data
df_train_raw = pd.read_csv('../data/raw/train.csv')
df_test_raw = pd.read_csv('../data/raw/test.csv')
df = df_train_raw.append(df_test_raw)
df = df.reset_index()

# Variables
target = 'label'
maincol = 'tweet'
mc = maincol + '_clean'
mcl = maincol + '_lst_clean'
mce = mc + '_emoji'
mcle = mcl + '_emoji'

# ==================== Useful functions ==============
def parallelize_dataframe(df, func):
    ncores = mp.cpu_count()
    df_split = np.array_split(df, ncores)
    pool = mp.Pool(ncores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

from urllib.parse import urlparse
def is_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    
#================== Text processing =================
def process_text(text):
    """
    Do a basic text processing.

    Parameters
    -----------
    text : string

    Returns
    --------
    This function returns pandas series having one list
    with clean text.
    1: split combined text
    2: lowercase
    3: expand apostrophes
    4: remove punctuation
    5: remove digits
    6: remove repeated substring
    7: remove stop words
    8: lemmatize

    Example:
    ========
    import re
    import string
    from nltk.corpus import stopwords
    import nltk
    
    text = "I'm typing text2num! areYou ? If yesyes say yes pals!"
    process_text(text)
    # ['typing', 'textnum', 'yes', 'say', 'yes', 'pal']

    """
    s = pd.Series([text])
    
    # step: Split combined words areYou ==> are You
    #s = s.apply(lambda x: re.sub(r'([a-z])([A-Z])',r'\1 \2',x))

    # step: lowercase
    s = s.str.lower()
    
    # step: remove ellipsis
    #s = s.str.replace(r'(\w)\u2026+',r'\1',regex=True)
    s = s.str.replace(r'…+',r'')

    # step: remove url
    #s = s.str.replace('http\S+|www.\S+', '', case=False)
    s = pd.Series([' '.join(y for y in x.split() if not is_url(y)) for x in s])

    # step: expand apostrophes
    map_apos = {
        "you're": 'you are',
        "i'm": 'i am',
        "he's": 'he is',
        "she's": 'she is',
        "it's": 'it is',
        "they're": 'they are',
        "can't": 'can not',
        "couldn't": 'could not',
        "don't": 'do not',
        "don;t": 'do not',
        "didn't": 'did not',
        "doesn't": 'does not',
        "isn't": 'is not',
        "wasn't": 'was not',
        "aren't": 'are not',
        "weren't": 'were not',
        "won't": 'will not',
        "wouldn't": 'would not',
        "hasn't": 'has not',
        "haven't": 'have not',
        "what's": 'what is',
        "that's": 'that is',
    }

    sa = pd.Series(s.str.split()[0])
    sb = sa.map(map_apos).fillna(sa)
    sentence = sb.str.cat(sep=' ')
    s = pd.Series([sentence])
    
    # step: expand shortcuts
    shortcuts = {'u': 'you', 'y': 'why', 'r': 'are',
                 'doin': 'doing', 'hw': 'how',
                 'k': 'okay', 'm': 'am', 'b4': 'before',
                 'idc': "i do not care", 'ty': 'thankyou',
                 'wlcm': 'welcome', 'bc': 'because',
                 '<3': 'love', 'xoxo': 'love',
                 'ttyl': 'talk to you later', 'gr8': 'great',
                 'bday': 'birthday', 'awsm': 'awesome',
                 'gud': 'good', 'h8': 'hate',
                 'lv': 'love', 'dm': 'direct message',
                 'rt': 'retweet', 'wtf': 'hate',
                 'idgaf': 'hate','irl': 'in real life',
                 'yolo': 'you only live once'}

    sa = pd.Series(s.str.split()[0])
    sb = sa.map(shortcuts).fillna(sa)
    sentence = sb.str.cat(sep=' ')
    s = pd.Series([sentence])

    # step: remove punctuation
    s = s.str.translate(str.maketrans(' ',' ',
                                        string.punctuation))
    # step: remove digits
    s = s.str.translate(str.maketrans(' ', ' ', '\n'))
    s = s.str.translate(str.maketrans(' ', ' ', string.digits))

    # step: remove repeated substring yesyes ==> yes
    s = s.str.replace(r'(\w+)\1',r'\1',regex=True)

    # step: remove stop words
    stop = set(stopwords.words('English'))
    extra_stop_words = ['...']
    stop.update(extra_stop_words) # inplace operation
    s = s.str.split()
    s = s.apply(lambda x: [I for I in x if I not in stop])

    # step: convert word to base form or lemmatize
    lemmatizer = nltk.stem.WordNetLemmatizer()
    s = s.apply(lambda lst: [lemmatizer.lemmatize(word) 
                               for word in lst])

    return s.to_numpy()[0]

def add_features(df):
    df[mcl] = df[maincol].apply(process_text)
    df[mc] = df[mcl].str.join(' ')
    df['hashtags_lst'] = df[maincol].str.findall(r'#.*?(?=\s|$)')
    
    #df['hashtags'] = df[maincol].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
    
    df['hashtags'] = df['hashtags_lst'].str.join(' ')

    return df

print("Creating clean tweet and hashtags ...")
df = parallelize_dataframe(df, add_features)

#======================= Text Feature Generation =====
def create_text_features(df):
    # total
    df['total_length'] = df[maincol].apply(len)

    # num of word and sentence
    df['num_words'] = df[maincol].apply(lambda x: len(x.split()))

    df['num_sent']=df[maincol].apply(lambda x: 
                                len(re.findall("\n",str(x)))+1)

    df['num_unique_words'] = df[maincol].apply(
        lambda x: len(set(w for w in x.split())))

    df["num_words_title"] = df[maincol].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

    df['num_uppercase'] = df[maincol].apply(
        lambda x: sum(1 for c in x if c.isupper()))

    # num of certain characters ! ? . @
    df['num_exclamation_marks'] = df[maincol].apply(lambda x: x.count('!'))

    df['num_question_marks'] = df[maincol].apply(lambda x: x.count('?'))

    df['num_punctuation'] = df[maincol].apply(
        lambda x: sum(x.count(w) for w in '.,;:'))

    df['num_symbols'] = df[maincol].apply(
        lambda x: sum(x.count(w) for w in '*&$%'))
    
    df['num_digits'] = df[maincol].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

    # average
    df["avg_word_len"] = df[maincol].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    
    df['avg_uppercase'] = df.apply(
        lambda row: float(row['num_uppercase'])/float(row['total_length']),
                                    axis=1)

    df['avg_unique'] = df['num_unique_words'] / df['num_words']
    
    return df

print("Adding Text features ...")
df = parallelize_dataframe(df, create_text_features)

#===================== Manipulating emoticons and emojis
from emojis import *
from emoticons import *

def convert_emoticons(text):
    for emot in EMOTICONS:
        text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
    return text

def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = re.sub(r'('+emot+')', "_".join(UNICODE_EMO[emot].replace(",","").replace(":","").split()), text)
    return text

def process_text_emoji(text):
    """
    Do a basic text processing.

    Parameters
    -----------
    text : string
        
    Returns
    --------
    This function returns pandas series having one list
    with clean text.
    1: split combined text
    2: lowercase
    3: expand apostrophes
    4: remove punctuation
    5: remove digits
    6: remove repeated substring
    7: remove stop words
    8: lemmatize

    Example:
    ========
    import re
    import string
    from nltk.corpus import stopwords
    import nltk
    
    text = "I'm typing text2num! areYou ? If yesyes say yes pals!"
    process_text(text)
    # ['typing', 'textnum', 'yes', 'say', 'yes', 'pal']

    """
    s = pd.Series([text])
    
    # step: expand emoticons and emojis
    s = s.apply(convert_emoticons)
    s = s.apply(convert_emojis)

    # step: Split combined words areYou ==> are You
    #s = s.apply(lambda x: re.sub(r'([a-z])([A-Z])',r'\1 \2',x))

    # step: lowercase
    s = s.str.lower()
    
    # step: remove ellipsis
    #s = s.str.replace(r'(\w)\u2026+',r'\1',regex=True)
    s = s.str.replace(r'…+',r'')

    # step: remove url
    #s = s.str.replace('http\S+|www.\S+', '', case=False)
    s = pd.Series([' '.join(y for y in x.split() if not is_url(y)) for x in s])

    # step: expand apostrophes
    map_apos = {
        "you're": 'you are',
        "i'm": 'i am',
        "he's": 'he is',
        "she's": 'she is',
        "it's": 'it is',
        "they're": 'they are',
        "can't": 'can not',
        "couldn't": 'could not',
        "don't": 'do not',
        "don;t": 'do not',
        "didn't": 'did not',
        "doesn't": 'does not',
        "isn't": 'is not',
        "wasn't": 'was not',
        "aren't": 'are not',
        "weren't": 'were not',
        "won't": 'will not',
        "wouldn't": 'would not',
        "hasn't": 'has not',
        "haven't": 'have not',
        "what's": 'what is',
        "that's": 'that is',
    }

    sa = pd.Series(s.str.split()[0])
    sb = sa.map(map_apos).fillna(sa)
    sentence = sb.str.cat(sep=' ')
    s = pd.Series([sentence])
    
    # step: expand shortcuts
    shortcuts = {'u': 'you', 'y': 'why', 'r': 'are',
                 'doin': 'doing', 'hw': 'how',
                 'k': 'okay', 'm': 'am', 'b4': 'before',
                 'idc': "i do not care", 'ty': 'thankyou',
                 'wlcm': 'welcome', 'bc': 'because',
                 '<3': 'love', 'xoxo': 'love',
                 'ttyl': 'talk to you later', 'gr8': 'great',
                 'bday': 'birthday', 'awsm': 'awesome',
                 'gud': 'good', 'h8': 'hate',
                 'lv': 'love', 'dm': 'direct message',
                 'rt': 'retweet', 'wtf': 'hate',
                 'idgaf': 'hate','irl': 'in real life',
                 'yolo': 'you only live once'}

    sa = pd.Series(s.str.split()[0])
    sb = sa.map(shortcuts).fillna(sa)
    sentence = sb.str.cat(sep=' ')
    s = pd.Series([sentence])

    # step: remove punctuation
    s = s.str.translate(str.maketrans(' ',' ',
                                        string.punctuation))
    # step: remove digits
    s = s.str.translate(str.maketrans(' ', ' ', '\n'))
    s = s.str.translate(str.maketrans(' ', ' ', string.digits))

    # step: remove repeated substring yesyes ==> yes
    s = s.str.replace(r'(\w+)\1',r'\1',regex=True)

    # step: remove stop words
    stop = set(stopwords.words('English'))
    extra_stop_words = ['...']
    stop.update(extra_stop_words) # inplace operation
    s = s.str.split()
    s = s.apply(lambda x: [I for I in x if I not in stop])

    # step: convert word to base form or lemmatize
    lemmatizer = nltk.stem.WordNetLemmatizer()
    s = s.apply(lambda lst: [lemmatizer.lemmatize(word) 
                               for word in lst])

    return s.to_numpy()[0]

def add_features_emoji(df):
    # we need to remove url first
    df[mcle] = df[maincol].str.replace('http\S+|www.\S+', '', case=False)
    df[mcle] = df[mcle].apply(process_text_emoji)
    df[mce] = df[mcle].str.join(' ')

    return df

print("Adding Emoticons and emoji features ...")
df = parallelize_dataframe(df, add_features_emoji)

#===================== Save clean data =========================
df.to_csv('../data/processed/df_combined_clean.csv',index=False)

time_taken = time.time() - time_start
m,s = divmod(time_taken,60)
print(f"Data cleaning finished in {m} min {s:.2f} sec.")

# Data cleaning finished in 12.0 min 6.19 sec.
