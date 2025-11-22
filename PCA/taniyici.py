import cv2
import numpy as np
import os

cascade_path = '../Cascade/haarcascade_frontalface_default.xml'
trainer_path = 'trainer/trainer.npy'  # PCA modeli dosyası

detector = cv2.CascadeClassifier(cascade_path)

if not os.path.exists(trainer_path):
    print(f"Model dosyası '{trainer_path}' bulunamadı. Lütfen önce eğitimi çalıştırın.")
    exit()

# Model verilerini yükle
face_data = np.load(trainer_path, allow_pickle=True).item()
pca = face_data['pca']
scaler = face_data['scaler']
faces_pca = face_data['faces_pca']
ids = face_data['ids']

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
        test_image = face_img.flatten()

        try:
            # Test görüntüsünü normalize et ve PCA alt uzayına indir
            test_image_scaled = scaler.transform([test_image])
            test_image_pca = pca.transform(test_image_scaled)

            # Öklid mesafesi ile en yakın ID'yi bul
            best_match = None
            best_score = float('inf')

            for i, face_pca in enumerate(faces_pca):
                score = np.linalg.norm(face_pca - test_image_pca)
                if score < best_score:
                    best_score = score
                    best_match = ids[i]

            # Güven oranı hesaplama ve tanımlama
            confidence = 100 - best_score  # Örneğin tersine çevirerek kullanabilirsiniz
            name = best_match if confidence > 50 else "Bilinmiyor"
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
