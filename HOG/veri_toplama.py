import cv2
import os

cam = cv2.VideoCapture(0)

# kamera çözünürlüğünü yüksek ayarlama işlemi
cam.set(3, 720)  # genişlik
cam.set(4, 480)  # yükseklik

face_detector = cv2.CascadeClassifier("../Cascade/haarcascade_frontalface_default.xml")

face_id = input('\n Bir isim giriniz ==>  ')

data_dir = "../data/" + face_id
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print("\n Yüz çekim işlemi başlatılıyor. Lütfen yüzünüzü farklı yönlere çevirin...")

count = 0
directions = ["düz", "sağa", "sola", "yukarı", "aşağı"]

for direction in directions:
    print(f"\nLütfen yüzünüzü {direction} çevirin ve birkaç saniye bekleyin...")
    frames_captured = 0  # her yön için yakalanan kare sayısının takibi
    while frames_captured < 20:  # her yön için 20 görüntü kaydedilecek(önce 10 görüntü ile denendi ama daha yüksek veri daha iyi sonuç veriyor)
        ret, img = cam.read()

        if not ret:
            print("Kamera görüntüsü alınamadı.")
            break

        # görüntüyü yatay olarak çevirme işlemi (kamerada ayna etkisi için)
        img = cv2.flip(img, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # yüzleri algılama işlemi burada gerçekleşiyor
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # yüzü ortalamaya çalışrak daha iyi dikdörgen içine almayı sağlıyor 
            x = max(x - 10, 0)
            y = max(y - 10, 0)
            w = min(w + 20, img.shape[1] - x)
            h = min(h + 20, img.shape[0] - y)

            # yüzün etrafına dikdörtgen çizme işlemi
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # yüzün iç kısmını keserek sadece yüzü veri olarak alıyor
            face_img = gray[y:y + h, x:x + w]

            face_img = cv2.resize(face_img, (200, 200))

            # veriyi kaydetme
            count += 1
            frames_captured += 1
            cv2.imwrite(os.path.join(data_dir, str(count) + ".jpg"), face_img)

        cv2.imshow('image', img)

        cv2.waitKey(500) 

        # 'esc' tuşuna basıldığında döngüyü kır
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

cam.release()
cv2.destroyAllWindows()

print("\n Yüz yakalama tamamlandı, {} görüntü kaydedildi.".format(count))
