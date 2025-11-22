import cv2
import numpy as np
from PIL import Image
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data_path = '../data'
trainer_path = 'trainer/trainer.npy'
cascade_path = '../Cascade/haarcascade_frontalface_default.xml'

detector = cv2.CascadeClassifier(cascade_path)

# Veritabanındaki yüzleri ve ID'leri yüklemek için bir fonksiyon
def getImagesAndLabels(path):
    faceSamples = []
    ids = []

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)

        if os.path.isdir(folder_path):
            id = folder_name

            try:
                imagePaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]

                for imagePath in imagePaths:
                    try:
                        PIL_img = Image.open(imagePath).convert('L')
                        img_numpy = np.array(PIL_img, 'uint8')

                        faces = detector.detectMultiScale(img_numpy, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

                        for (x, y, w, h) in faces:
                            face = img_numpy[y:y + h, x:x + w]
                            face_resized = cv2.resize(face, (100, 100))  # Sabit boyuta yeniden boyutlandır
                            faceSamples.append(face_resized.flatten())
                            ids.append(id)
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
            # Yüz verilerini normalize et
            faces = np.array(faces)
            ids = np.array(ids)
            scaler = StandardScaler()
            faces_scaled = scaler.fit_transform(faces)

            # PCA ile yüzleri alt uzaya indirgeme
            n_components = min(len(faces), 95)  # En fazla örnek sayısı kadar bileşen seçilebilir
            pca = PCA(n_components=n_components)  # Bileşen sayısını dinamik olarak ayarla
            faces_pca = pca.fit_transform(faces_scaled)

            # PCA bileşenleri, ölçekleyici ve ID'leri kaydet
            data = {
                "pca": pca,
                "scaler": scaler,
                "faces_pca": faces_pca,
                "ids": ids
            }
            np.save(trainer_path, data)
            print(f"Model başarıyla '{trainer_path}' dosyasına yazıldı.")
        except Exception as e:
            print(f"Hata: {e}")
    else:
        print("Eğitim verisi yok. Yüzler veya ID'ler bulunamadı.")


def recognize_face(test_image):
    if not os.path.exists(trainer_path):
        print("Model dosyası bulunamadı. Önce modeli eğitin.")
        return

    face_data = np.load(trainer_path, allow_pickle=True).item()
    pca = face_data["pca"]
    scaler = face_data["scaler"]
    faces_pca = face_data["faces_pca"]
    ids = face_data["ids"]

    try:
        # Test görüntüsünü griye çevir ve normalize et
        test_image = test_image.flatten()
        test_image_scaled = scaler.transform([test_image])

        # Test görüntüsünü PCA alt uzayına indir
        test_image_pca = pca.transform(test_image_scaled)

        # Öklid mesafesi ile en yakın ID'yi bul
        best_match = None
        best_score = float("inf")

        for i, face_pca in enumerate(faces_pca):
            score = np.linalg.norm(face_pca - test_image_pca)
            if score < best_score:
                best_score = score
                best_match = ids[i]

        return best_match, best_score

    except Exception as e:
        print(f"Hata: {e}")
        return None, None

train_model()
