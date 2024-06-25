
# from joblib import dump

# # model
# dump(xgb_model, 'model_filenameG.joblib')


# #save model

from api_model_red_neuronal import predict_url
from joblib import load


model = load('modelo_xbg.joblib')

label_mapping = {'benign': 0, 'defacement': 1, 'phishing': 2, 'malware': 3}
# Probar con ejemplos en vivo
test_urls = [
     "https://www.linkedin.com/feed/?highlightedUpdateType=SHARED_BY_YOUR_NETWORK&highlightedUpdateUrn=urn%3Ali%3Aactivity%3A7202827684500930560",
     "http://117.207.242.228:60378/i",
     "mp3raid.com/music/krizz_kaliko.html",
     "br-icloud.com.br",
     "https://leerolymp.com/capitulo/93966/comic-tu-talento-ahora-es-mio13424"
]


for url in test_urls:
     prediction = predict_url(url, model, label_mapping)
     print(f"URL: {url} -> Prediction: {prediction}")

'''

url= input("Introduce la URL: ")
prediction = predict_url(url, model, label_mapping)
print(f"URL: {url} -> Prediction: {prediction}")
'''

