import cv2
import numpy as np
from PIL import Image
import os

data_path = '../data'
trainer_path = 'trainer/trainer.npy'
cascade_path = '../Cascade/haarcascade_frontalface_default.xml'


detector = cv2.CascadeClassifier(cascade_path)

# veritabanındaki yüzleri ve ID'leri yüklemek için bir fonksiyon
def getImagesAndLabels(path):
    faceSamples = []
    ids = []

    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)

        if os.path.isdir(folder_path):
            # ID olarak klasör adını kullan
            id = folder_name

            try:
                imagePaths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')]

                for imagePath in imagePaths:
                    try:
                        # resmi gri tonlamaya çevir
                        PIL_img = Image.open(imagePath).convert('L')
                        img_numpy = np.array(PIL_img, 'uint8')

                        # yüzleri tespit et
                        faces = detector.detectMultiScale(img_numpy, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

                        for (x, y, w, h) in faces:
                            faceSamples.append(img_numpy[y:y + h, x:x + w])
                            ids.append(id)  # ID'yi ekle
                    except Exception as e:
                        print(f"Error processing image {imagePath}: {e}")

            except Exception as e:
                print(f"Error processing folder {folder_path}: {e}")

    return faceSamples, ids

def calculate_histogram(image):
    # histogram oluşturma ve normalize etme işlemi
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_histograms(hist1, hist2):
    # histogram karşılaştırması yapılıyor (korelasyon yöntemi)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def train_model():
    if not os.path.exists('trainer'):
        os.makedirs('trainer')

    print("\nYüzler taranıyor. Birkaç saniye sürecek, bekleyin...")
    faces, ids = getImagesAndLabels(data_path)

    if len(faces) > 0 and len(ids) > 0:
        try:
            # IDs dizisini düzenle ve yüzlerin histogramlarını hesapla
            histograms = []
            for face in faces:
                face_hist = calculate_histogram(face)
                histograms.append(face_hist)

            # histogramları ve ID'leri bir sözlükte sakla
            data = {
                "histograms": histograms,
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

    test_hist = calculate_histogram(test_image)

    best_match = None
    best_score = -1

    # tüm ID'ler ve histogramlar arasında karşılaştırma yaparak doğru veriyi bulma işlemi
    for uid, histograms in face_data.items():
        for hist in histograms:
            score = compare_histograms(test_hist, hist)
            if score > best_score:
                best_score = score
                best_match = uid

    return best_match, best_score

train_model()
