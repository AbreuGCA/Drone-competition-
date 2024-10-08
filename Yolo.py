import cv2
from ultralytics import YOLO

# Inicializa o modelo YOLOv8
model = YOLO("yolov8n.pt")

# Inicializa a webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Captura frame por frame
    success, frame = cap.read()
    
    if success:
        # Realiza a detecção
        results = model(frame)
        
        # Processa os resultados
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Verifica se a classe detectada é pessoa (classe 0)
                if box.cls[0] == 0:
                    # Extrai as coordenadas da caixa
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Desenha a caixa delimitadora
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Adiciona o rótulo com a confiança
                    conf = float(box.conf[0])
                    label = f"Pessoa: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Mostra o resultado
        cv2.imshow("YOLOv8 Detecção de Pessoas", frame)
        
        # Verifica se a tecla 'q' foi pressionada para sair
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()