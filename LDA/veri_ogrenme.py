import cv2
import numpy as np
import os
from PIL import Image

# Veri yolları
data_path = '../data'
trainer_path = 'trainer/trainer.yml'  # Modelin kaydedileceği dosya
cascade_path = '../Cascade/haarcascade_frontalface_default.xml'

detector = cv2.CascadeClassifier(cascade_path)

# Veritabanındaki yüzleri ve ID'leri yüklemek için bir fonksiyon
def getImagesAndLabels(path):
    faceSamples = []
    ids = []
    folder_to_id = {}  # Klasör adlarını tamsayıya eşleyecek bir sözlük

    current_id = 0  # Başlangıç ID'si

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)

        if os.path.isdir(folder_path):
            # Klasör adını bir ID'ye eşle
            if folder_name not in folder_to_id:
                folder_to_id[folder_name] = current_id
                current_id += 1  # Yeni bir ID oluştur

            folder_id = folder_to_id[folder_name]

            try:
                imagePaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]

                for imagePath in imagePaths:
                    try:
                        # Resmi gri tonlamaya çevir
                        PIL_img = Image.open(imagePath).convert('L')
                        img_numpy = np.array(PIL_img, 'uint8')

                        # Yüzleri tespit et
                        faces = detector.detectMultiScale(img_numpy, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

                        for (x, y, w, h) in faces:
                            face = img_numpy[y:y + h, x:x + w]

                            # Yüzleri aynı boyutta yeniden boyutlandır (örneğin, 100x100)
                            face_resized = cv2.resize(face, (100, 100))

                            faceSamples.append(face_resized)
                            ids.append(folder_id)  # Klasör ID'sini ekle
                    except Exception as e:
                        print(f"Error processing image {imagePath}: {e}")

            except Exception as e:
                print(f"Error processing folder {folder_path}: {e}")

    return faceSamples, ids


def train_model():
    if not os.path.exists('trainer'):
        os.makedirs('trainer')

    print("\nYüzler taranıyor. Birkaç saniye sürecek, bekleyin...")
    faces, ids = getImagesAndLabels(data_path)

    if len(faces) > 0 and len(ids) > 0:
        try:
            # FisherFaceRecognizer (LDA tabanlı) modelini oluştur
            recognizer = cv2.face.FisherFaceRecognizer_create()  # LDA tabanlı model
            recognizer.train(faces, np.array(ids))  # Yüzleri eğit

            # Modeli kaydet
            recognizer.save(trainer_path)  # .yml uzantısı genellikle doğru kaydetme uzantısıdır
            print(f"Model başarıyla '{trainer_path}' dosyasına yazıldı.")
        except Exception as e:
            print(f"Hata: {e}")
    else:
        print("Eğitim verisi yok. Yüzler veya ID'ler bulunamadı.")


train_model()
