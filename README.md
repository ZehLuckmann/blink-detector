# BlinkDetector

Neste projeto, mostro como construir um detector do piscar de olhos usando OpenCV, Python e dlib.

O primeiro passo é realizar a detecção da referência facial para localizar os olhos em uma determinada moldura de um vídeo.

Uma vez que temos os marcos faciais para ambos os olhos, calculamos a relação de aspecto do olho para cada olho, o que nos dá um valor único, relacionando as distâncias entre os pontos de referência vertical do olho e as distâncias entre os pontos de referência horizontais.

Uma vez que temos a proporção de aspecto do olho, podemos definir um limiar para determinar se uma pessoa está piscando - a proporção de aspecto do olho permanecerá aproximadamente constante quando os olhos estiverem abertos e então se aproximará rapidamente de zero durante um piscar, então aumentará novamente à medida que o olho se abrir.
