import cv2
#import numpy as np
import time

# CORES DAS CLASSES
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
# CARREGA AS CLASSES
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
    # class_names = [line.strip() for line in f]
    # print(class_names)  # Teste para saber se esta acessando a lista

# CAPTURA DO VIDEO
cap = cv2.VideoCapture("GarotaCavalo.gif")

# CARREGA OS PESOS DA REDE NEURAL

# CONSOME MAIS RECURSOS DO SISTEMA
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# MAIS LEVE POREM MENOS EFICAS
#net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

# SETANDO OS PARAMETROS DA REDE NEURAL
model = cv2.dnn_DetectionModel(net)
# MAIS LEVE
#model.setInputParams(size=(416, 416), scale=1/255) 

# MAIS RECURSOS 
model.setInputParams(size=(608, 608), scale=1/255) 

# LENDO OS FRAMES DO VIDEO
while True:

    # CAPTURA DO FRAME
    _, frame = cap.read()

    # COMEÇO DA CONTAGEM DOS MS
    start = time.time()

    # DETECÇÃO
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    # FIM DA CONTAGEM DOS MS
    end = time.time()

    # PERCORRER TODAS AS DETECCÇÕES
    for (classid, score, box) in zip(classes, scores, boxes):

        # GERANDO UMA COR PARA A CLASSE
        color = COLORS[int(classid) % len(COLORS)]

        # PEGANDO O NOME DA CLASSE PELO ID E O SEU SCORE DE ACURACIA
        nome = {class_names[classid]}
        pontos = '{0:.4g}'.format(score)
        label = f"{nome} : {pontos}"

        if score > (0.4):
            # DESENHANDO A BOX DE DETECÇÃO
            cv2.rectangle(frame, box, color, 2)

            # ESCREVENDO O NOME DA CLASSE EM CIMA DA BOX DO OBJETO
            cv2.putText(frame, label, ((box[0], box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # CALCULANDO O TEMPO QUE LEVOU PARA FAZER A DETECÇÃO
    fps_label = f"FPS: {round((1.0/(end - start)),2)}"

    # ESCREVENDO O FPS NA IMAGEM
    cv2.putText(frame, fps_label, (0, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(frame, fps_label, (0, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    # MOSTRANDO A IMAGEM
    cv2.imshow("Detections", frame)

    # ESPERA DA RESPOSTA
    if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
        break

# LIBERAÇÃO DA CAMERA E DESTROI TODAS AS JANELAS
cap.release()
cv2.destroyAllWindows()
