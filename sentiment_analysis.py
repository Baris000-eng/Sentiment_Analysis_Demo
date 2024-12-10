import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

data = {
    'text': [
        "Bu film harikaydı, çok sevdim!",
        "Kötü bir deneyimdi, hiç beğenmedim.",
        "Gerçekten çok güzel bir hikaye.",
        "Berbat bir performans, hiç izlemek istemem.",
        "Efsanevi bir yapım, kesinlikle tavsiye ederim!",
        "Bu kitap harikaydı, mutlaka okuyun.",
        "Son derece sıkıcı bir film, zaman kaybı.",
        "Müthiş bir deneyim, tekrar izlerim.",
        "İnanılmaz bir anlatım, çok etkileyici.",
        "Beni derinden etkiledi, mükemmel bir yapım.",
        "Berbat bir senaryo, asla izlemeyin.",
        "Görselliği çok iyiydi, ama hikaye zayıf.",
        "Çocuklar için eğlenceli bir film.",
        "Tam bir hayal kırıklığı, beklentilerimi karşılamadı.",
        "Duygusal bir film, gözyaşlarımı tutamadım.",
        "Korkunç bir filmdi, hiç beğenmedim.",
        "Sürükleyici bir hikaye, çok beğendim.",
        "Eğlenceli bir komedi, kahkahalarla izledim.",
        "Gerçekten harika bir film, herkes izlemeli.",
        "Daha iyi olabilirdi, ama yine de güzeldi.",
        "Süper bir filmdi.",
        "Çok güzel bir filmdi.",
        "Güzel bir filmdi.",
        "Müthiş bir filmdi.",
        "Çok kötüydü.",
        "Çok igrençti.",
        "Çok rezil bir durumdu.",
        "Kötüydü",
        "İgrençti",
        "Çok güzeldi."
    ],
    'label': [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1]  # Duygu durumları
}

df = pd.DataFrame(data) 

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Vocabulary oluşturuyor
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)  
X_test_vectorized = vectorizer.transform(X_test) 

# Modeli oluşturuyoruz
model = LogisticRegression(random_state=42)


n_epochs = 200
for epoch in range(n_epochs):
    model.fit(X_train_vectorized, y_train)

# Test verisi üzerinde tahmin yapma
y_pred = model.predict(X_test_vectorized)

# Model başarımları
accuracy_test = accuracy_score(y_test, y_pred)
f1_test = f1_score(y_test, y_pred)
recall_test = recall_score(y_test, y_pred)

print('Doğruluk oranı: ' + str(accuracy_test))
print('F1-Score değeri: ' + str(f1_test))
print('Recall değeri: ' + str(recall_test))

# Yeni bir metin için duygu analizi yapma 
new_text = ["Korkunç."]
new_text_vectorized = vectorizer.transform(new_text)  

probabilities = model.predict_proba(new_text_vectorized) 
print('Probabilities of predicting each class are: ' + str(probabilities))
prediction = model.predict(new_text_vectorized)
print("Yeni metin duygu durumu:", "Olumlu" if prediction[0] == 1 else "Olumsuz")
