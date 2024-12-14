import cv2
import numpy as np

# Haarcascade dosyasını yükleme
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Kamerayı başlatma
cap = cv2.VideoCapture(0)

def is_eye_closed(eye_region):
    # Göz kapalı mı kontrol etme (threshold yöntemi)
    _, threshold_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) == 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gözleri algılama
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

    for (x, y, w, h) in eyes:
        eye_region = gray[y:y+h, x:x+w]
        
        if is_eye_closed(eye_region):
            print("Gözler kapalı, çizim yapılmıyor.")
        else:
            _, threshold_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)

            # Gözbebeğini bulma
            contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            for cnt in contours:
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                cx, cy = int(cx), int(cy)
                if radius > 5:
                    eye_center = (x + cx, y + cy)
                    cv2.circle(frame, eye_center, int(radius), (0, 255, 0), 2)
                    
                    # Büyük mavi artı çizimi
                    cv2.line(frame, (eye_center[0] - 10, eye_center[1]), (eye_center[0] + 10, eye_center[1]), (255, 0, 0), 2)
                    cv2.line(frame, (eye_center[0], eye_center[1] - 10), (eye_center[0], eye_center[1] + 10), (255, 0, 0), 2)

                    print(f"Göz algılandı, konum: {eye_center}")
                    break

    # Çıktıyı gösterme
    cv2.imshow("Goz Takibi", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
