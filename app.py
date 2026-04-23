import cv2
import numpy as np

def processar_por_blocos(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold adaptativo para lidar com sombras na foto
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 1. IDENTIFICAR OS BLOCOS (As molduras das matérias)
    # Procuramos por contornos grandes que envolvam as questões
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blocos = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Filtra apenas retângulos grandes (os blocos de disciplinas)
        if w > 150 and h > 100:
            blocos.append((x, y, w, h))

    # Ordenar blocos: primeiro por Y (linha) e depois por X (coluna)
    # Isso garante que leia: Português, Matemática, Biologia... na ordem certa
    blocos.sort(key=lambda b: (b[1] // 50, b[0]))

    gabarito_final = {}
    questao_global = 1

    for (bx, by, bw, bh) in blocos:
        # Extrai a imagem apenas daquele bloco (ex: Matemática)
        roi_bloco = thresh[by:by+bh, bx:bx+bw]
        
        # 2. IDENTIFICAR AS LINHAS DENTRO DO BLOCO
        # Vamos contar quantos círculos/quadrados de opções existem por linha
        # e agrupar para saber que a questão X tem as opções A,B,C,D,E
        opcoes = []
        c_internos, _ = cv2.findContours(roi_bloco, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for ci in c_internos:
            ix, iy, iw, ih = cv2.boundingRect(ci)
            ar = iw / float(ih)
            # Filtra apenas os quadradinhos das alternativas
            if 15 < iw < 40 and 0.8 < ar < 1.2:
                opcoes.append((ix, iy, iw, ih, ci))
        
        # Ordena as opções de cima para baixo
        opcoes.sort(key=lambda o: o[1])

        # Agrupa de 5 em 5 para formar uma questão
        for i in range(0, len(opcoes), 5):
            linha_questao = opcoes[i:i+5]
            if len(linha_questao) < 5: continue
            
            # Ordena da esquerda para a direita (A, B, C, D, E)
            linha_questao.sort(key=lambda o: o[0])
            
            # Analisa qual está preenchida
            votos = []
            for (ox, oy, ow, oh, contorno) in linha_questao:
                mask = np.zeros(roi_bloco.shape, dtype="uint8")
                cv2.drawContours(mask, [contorno], -1, 255, -1)
                mask = cv2.bitwise_and(roi_bloco, roi_bloco, mask=mask)
                votos.append(cv2.countNonZero(mask))
            
            alternativas = ["A", "B", "C", "D", "E"]
            escolhida = alternativas[np.argmax(votos)]
            
            gabarito_final[str(questao_global)] = escolhida
            questao_global += 1

    return gabarito_final
