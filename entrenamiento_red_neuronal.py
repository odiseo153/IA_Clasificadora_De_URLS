import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re
import joblib
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tld import get_tld



def load_and_balance_data(filepath):
    df = pd.read_csv(filepath)
    df['url'] = df['url'].astype(str).fillna('')
    df['type'] = df['type'].fillna('')

    phishing_count = df[df['type'] == 'phishing'].shape[0]
    df_malware = df[df['type'] == 'malware'].sample(phishing_count, random_state=42)

    df_phishing = df[df['type'] == 'phishing']
    df_benign = df[df['type'] == 'benign']
    df_defacement = df[df['type'] == 'defacement']

    df_balanced = pd.concat([df_phishing, df_malware, df_benign, df_defacement])
    return df_balanced


def having_ip_address(url):
    return int(bool(re.search(r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.)', url)))

def abnormal_url(url):
    try:
        hostname = url.split('/')[2]
        return int(bool(re.search(re.escape(hostname), url)))
    except IndexError:
        return 0

def count_dot(url):
    return url.count('.')

def count_www(url):
    return url.count('www')

def count_atrate(url):
    return url.count('@')

def no_of_dir(url):
    return url.count('/')

def no_of_embed(url):
    return url.count('//')

def shortening_service(url):
    match = re.search(
        r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|qr\.ae|adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net',
        url)
    if match:
        return 1
    else:
        return 0

def count_https(url):
    return url.count('https')

def count_http(url):
    return url.count('http')

def count_per(url):
    return url.count('%')

def count_ques(url):
    return url.count('?')

def count_hyphen(url):
    return url.count('-')

def count_equal(url):
    return url.count('=')

def url_length(url):
    return len(url)

def hostname_length(url):
    try:
        return len(url.split('/')[2])
    except IndexError:
        return 0

def suspicious_words(url):
    return int('security' in url or 'confirm' in url or 'bank' in url)

def digit_count(url):
    digits = [i for i in url if i.isdigit()]
    return len(digits)

def letter_count(url):
    letters = [i for i in url if i.isalpha()]
    return len(letters)

def fd_length(url):
    try:
        return len(url.split('/')[3])
    except IndexError:
        return 0

def tld_length(tld):
    return len(tld) if tld else 0

# Función para crear DataFrame de características
def extract_features(df):
    return df['url'].apply(lambda x: pd.Series([
        having_ip_address(x),
        abnormal_url(x),
        count_dot(x),
        count_www(x),
        count_atrate(x),
        no_of_dir(x),
        no_of_embed(x),
        shortening_service(x),
        count_https(x),
        count_http(x),
        count_per(x),
        count_ques(x),
        count_hyphen(x),
        count_equal(x),
        url_length(x),
        hostname_length(x),
        suspicious_words(x),
        digit_count(x),
        letter_count(x),
        fd_length(x),
        tld_length(get_tld(x, fail_silently=True))
    ]))


def prepare_data(df):
    label_mapping = {'benign': 0, 'defacement': 1, 'phishing': 2, 'malware': 3}
    df_features = extract_features(df)
    df_features.columns = ['having_ip_address', 'abnormal_url', 'count_dot', 'count_www', 'count_atrate',
                           'no_of_dir', 'no_of_embed', 'shortening_service', 'count_https', 'count_http',
                           'count_per', 'count_ques', 'count_hyphen', 'count_equal', 'url_length',
                           'hostname_length', 'suspicious_words', 'digit_count', 'letter_count', 'fd_length', 'tld_length']
    df_features['type'] = df['type'].map(label_mapping)
    return df_features.drop(columns=['type']), df_features['type']



def create_dnn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))  # 4 clases de salida
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


df_balanced = load_and_balance_data('combined_csv.csv')

X, y = prepare_data(df_balanced)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


y_train_one_hot = to_categorical(y_train, num_classes=4)
y_test_one_hot = to_categorical(y_test, num_classes=4)


input_dim = X_train.shape[1]
dnn_model = create_dnn_model(input_dim)
dnn_model.fit(X_train, y_train_one_hot, epochs=20, batch_size=32, validation_split=0.2)


y_pred_prob = dnn_model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)


target_names = ['benign', 'defacement', 'phishing', 'malware']
print(classification_report(y_test, y_pred, target_names=target_names))

# Precisión del modelo
score = accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)

dnn_model.save("modelo_red_neuronal.h5")
