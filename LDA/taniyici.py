import cv2
import numpy as np
import os

trainer_path = 'trainer/trainer.yml'  # LDA modeli dosyası
cascade_path = '../Cascade/haarcascade_frontalface_default.xml'

detector = cv2.CascadeClassifier(cascade_path)

# Model verilerini yükle
if not os.path.exists(trainer_path):
    print(f"Model dosyası '{trainer_path}' bulunamadı. Lütfen önce eğitimi çalıştırın.")
    exit()

recognizer = cv2.face.FisherFaceRecognizer_create()
recognizer.read(trainer_path)

# Kamerayı başlat
camera = cv2.VideoCapture(0)

print("Kamera açıldı. Tanıma işlemi başlatılıyor. Çıkmak için 'q' tuşuna basın.")

while True:
    ret, img = camera.read()
    if not ret:
        print("Kameradan görüntü alınamadı!")
        break

    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (100, 100))  # Eğitime uygun boyuta getir

        try:
            # Test görüntüsünü normalize et
            label, confidence = recognizer.predict(face_img)

            # Güven oranı hesaplama ve tanımlama
            confidence = 100 - confidence  # Örneğin tersine çevirerek kullanabilirsiniz
            name = f"ID: {label}" if confidence > 50 else "Bilinmiyor"
            print(f"Tanımlanan: {name}, Güven oranı: {confidence:.2f}%")

            # Yüz etrafına dikdörtgen çizdirme ve üzerine adı yazdırma işlemi
            color = (0, 255, 0) if name != "Bilinmiyor" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        except Exception as e:
            print(f"Tanıma işlemi sırasında hata: {e}")

    cv2.imshow('Tanima', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
