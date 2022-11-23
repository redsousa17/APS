import cv2
import numpy as np
import time

#-- CORES
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

#--- CARREGA AS CLASSES COM OS NOMES DOS OBJETOS (lista coco.names na pasta)
class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
    # print(class_names)  # Teste para saber se esta acessando a lista

#--- CAPTURA DO VIDEO (Arquivo ou WebCan)
cap = cv2.VideoCapture("walking.mp4")

#--- CARREGA OS PESOS DA REDE NEURAL 
#--- Consome MAIS recursos do Sistema (Mais Eficiente)
#net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg") 
#--- Consome MENOS recursos do Sistema (Menos Eficiente)
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg") 

#--- SETANDO OS PARAMETROS DA REDE NEURAL
model = cv2.dnn_DetectionModel(net)
#--- Consome MAIS recursos do Sistema (Mais Eficiente)
#model.setInputParams(size=(608, 608), scale=1/255) 
#--- Consome MENOS recursos do Sistema (Menos Eficiente)
model.setInputParams(size=(416, 416), scale=1/255) 

#--- LAÇO QUE LE FRAME A FRAME DO VIDEO
while True:

    #--- CAPTURA DO FRAME
    _, frame = cap.read()

    #--- COMEÇO DA CONTAGEM DO TEMPO PARA PROCESSAR
    start = time.time()

    #--- DETECÇÃO, BUSCA NO FRAME OBJETOS CATALOGADOS NA REDE NURAL
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    #--- FIM DA CONTAGEM TEMPO PARA PROCESSAR
    end = time.time()

    #--- PERCORRER TODAS AS DETECCÇÕES
    for (classid, score, box) in zip(classes, scores, boxes):

        #--- GERANDO UMA COR PARA A CLASSS (Cada objeto tem uma cor unica)
        color = COLORS[int(classid) % len(COLORS)]

        #--- PEGANDO O NOME DA CLASSE PELO ID E O SEU SCORE DE ACURACIA
        nome = {class_names[classid]}
        pontos = '{0:.4g}'.format(score)
        label = f"{nome} : {pontos}"

        #--- CLASSIFICA A PONTUAÇÂO E GARANTE QUE SOMENTE OBJETOS COM MAIS DE 0.4 SEJA EXIBIDO (Garante mais acertividade)
        if score > (0.5): 
            #--- DESENHANDO A BOX DE DETECÇÃO
            cv2.rectangle(frame, box, color, 2)

            #--- ESCREVENDO O NOME DA CLASSE EM CIMA DA BOX DO OBJETO
            #--- Borda
            cv2.putText(frame, label, ((box[0], box[1] - 10)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2, 255)
            #--- Preenchimento
            cv2.putText(frame, label, ((box[0], box[1] - 10)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, 255)

    #-- CALCULANDO O TEMPO QUE LEVOU PARA FAZER A DETECÇÃO
    fps_label = f"FPS: {round((1.0/(end - start)),2)}"

    #--- ESCREVENDO O FPS NA IMAGEM
    #--- Borda
    cv2.putText(frame, fps_label, (0, 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2, 255)
    #--- Preenchimento
    cv2.putText(frame, fps_label, (0, 25),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, 255)

    #--- JANELA DE DETECÇÃO
    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

    #--- MOSTRANDO A IMAGEM
    cv2.imshow("Detection", frame)

    #--- ESPERA DA RESPOSTA (Encerra a Execução apertando 'q' ou 'Q')
    if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
        break

#--- LIBERAÇÃO DA CAMERA E DESTROI TODAS AS JANELAS
cap.release()
cv2.destroyAllWindows()
