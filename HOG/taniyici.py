import cv2
import numpy as np
import os

cascade_path = '../Cascade/haarcascade_frontalface_default.xml'
histogram_data_path = 'trainer/trainer.npy' # histogram verilerinin kaydedildiği dosya


detector = cv2.CascadeClassifier(cascade_path) # yüz algılama işlemi

if not os.path.exists(histogram_data_path):
    print(f"Histogram verisi '{histogram_data_path}' bulunamadı. Lütfen önce eğitimi çalıştırın.")
    exit()

data = np.load(histogram_data_path, allow_pickle=True).item()
stored_histograms = data['histograms']
stored_ids = data['ids']

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
        face_img = cv2.resize(face_img, (200, 200))

        test_hist = cv2.calcHist([face_img], [0], None, [256], [0, 256])
        test_hist = cv2.normalize(test_hist, test_hist).flatten()

        # histogram karşılaştırması yapılıyor
        best_match_id = None
        best_score = -1
        for idx, stored_hist in enumerate(stored_histograms):
            score = cv2.compareHist(test_hist, stored_hist, cv2.HISTCMP_CORREL)
            if score > best_score:
                best_score = score
                best_match_id = stored_ids[idx]

        confidence = best_score * 100
        name = best_match_id if confidence > 50 else "Bilinmiyor"
        print(f"Tanımlanan: {name}, Güven oranı: {confidence:.2f}%")

        # yüz etrafına dikdörtgen çizdirme ve üzerine adı yazdırma işlemi
        color = (0, 255, 0) if name != "Bilinmiyor" else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Tanima', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
