#coding:utf-8
#Importa os pacotes necessários
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

MODELO_PREDITOR_ROSTO = "shape_predictor_68_face_landmarks.dat"
VIDEO = ""


def relacao_aspecto_olho(eye):
	#Calcular as distâncias euclidianas(geometria, em duas e três dimensões)
	#entre os dois marcos verticais dos olhos (x, y) - coordenadas
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	#Calcular as distâncias euclidianas(geometria, em duas e três dimensões)
	#entre o marco vertical dos olhos (x, y) - coordenadas
	C = dist.euclidean(eye[0], eye[3])

	#Calcular a proporção de aspecto do olho
	media_relacao_aspecto = (A + B) / (2.0 * C)
	return media_relacao_aspecto

#Defina duas constantes, uma para a relação de aspecto do olho para indicar
#quando piscar em seguida, uma segunda constante para o número de quadros
#consecutivos, o olho deve estar abaixo do limiar para contabilizar
LIMIAR_PISCADA = 0.3
NUMEROS_FRAMES_PISCADA = 3

#Inicializa os contadores de quadros e o número total de piscadas
CONTADOR = 0
TOTAL = 0

#Inicializa a detecção de rosto usando o dlib (baseado em HOG) e em seguida,
#cria a referência facial
print("[INFO] Carregando prévia de referência facial ...")
detector = dlib.get_frontal_face_detector()
preditor = dlib.shape_predictor(MODELO_PREDITOR_ROSTO)

# Pega os índices dos marcos faciais para o olho esquerdo e direito, respectivamente
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#Inicie o streaming do vídeo
print("[INFO] Iniciando o streaming de vídeo")
print("Escolha a forma de entrada")
print("[1] Câmera")
print("[2] Vídeo")
opcaoVideo = input("Selecione uma opção")

if opcaoVideo == 2:
	VIDEO = input("Informe o caminho do arquivo:")
	vs = FileVideoStream(VIDEO).start()	
	fileStream = False
else:
	vs = VideoStream(src=0).start() 
	fileStream = True

time.sleep(1.0)

# Loop por cada frame do vídeo
while True:
	# Se for um streaming de vídeo então precisa verificar se há
	# mais quadros no buffer para processar
	if fileStream and not vs.more():
		break

	# Converte o frame para um canal em escala de cinza
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detecta rostos no quadro em escala de cinza
	faces = detector(gray, 0)

	# loop pelas detecções do rosto
	for face in faces:
		# Determina os marcos faciais para a região do rosto e em seguida,
		# converte o marco facial (x, y) para uma matriz NumPy
		forma = preditor(gray, face)
		forma = face_utils.shape_to_np(forma)

		# Extraia as coordenadas do olho esquerdo e direito, depois use as
		# coordenadas para calcular a relação de aspecto do olho para ambos os olhos
		olho_esquerdo = forma[lStart:lEnd]
		olho_direito = forma[rStart:rEnd]
		relacao_aspecto_esquerdo = relacao_aspecto_olho(olho_esquerdo)
		relacao_aspecto_direito = relacao_aspecto_olho(olho_direito)

		# Média da relação da proporção para ambos os olhos
		media_relacao_aspecto = (relacao_aspecto_esquerdo + relacao_aspecto_direito) / 2.0

		# Calcular o convexo do olho esquerdo e direito
		contorno_olho_esquerdo = cv2.convexHull(olho_esquerdo)
		contorno_olho_direito = cv2.convexHull(olho_direito)
		cv2.drawContours(frame, [contorno_olho_esquerdo], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [contorno_olho_direito], -1, (0, 255, 0), 1)

		# Descubra se a relação de aspecto do olho está abaixo do limiar de intermitência
		# e em caso afirmativo, incremente o contador do quadro intermitente
		if media_relacao_aspecto < LIMIAR_PISCADA:
			CONTADOR += 1
		else:
			# Se os olhos estiverem fechados por um número suficiente de tempo
			# Então incrementa o número total de piscadas
			if CONTADOR >= NUMEROS_FRAMES_PISCADA:
				TOTAL += 1

			#Redefinir o contador do frame do olho
			CONTADOR = 0

		# Desenheao número total de pisca no quadro juntamente com a
		# relação de aspecto do olho calculado para o quadro
		cv2.putText(frame, "Piscadas: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "Int: {:.2f}".format(media_relacao_aspecto), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# Mostra o quadro
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Se a tecla 'q' foi pressionada, sai fora do loop
	if key == ord("q"):
		break

#Limpa a memória
cv2.destroyAllWindows()
vs.stop()
