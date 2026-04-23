def detectar_por_materias_preciso(image_file):
    # ESTRUTURA EXATA DO SEU MODELO (Nome da Matéria : Quantidade de Questões)
    estrutura_prova = [
        ("Língua Portuguesa", 10), ("Matemática", 10), ("Biologia", 4),
        ("Química", 4),           ("Física", 4),      ("História", 4),
        ("Geografia", 4),         ("Língua Inglesa", 2), ("Educação Física", 2),
        ("Arte", 2),              ("Sociologia", 2),   ("Filosofia", 2)
    ]
    
    filestr = image_file.read()
    nparr = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return {}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 1. Captura todos os quadradinhos de resposta da folha
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quadradinhos = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 18 < w < 48 and 0.7 < (w/h) < 1.3:
            quadradinhos.append((x, y, w, h, c))

    # 2. Ordena de cima para baixo
    quadradinhos.sort(key=lambda q: q[1])

    # 3. Agrupa em linhas de 5 (alternativas A,B,C,D,E)
    questoes_lidas = []
    for i in range(0, len(quadradinhos), 5):
        linha = quadradinhos[i:i+5]
        if len(linha) < 5: continue
        linha.sort(key=lambda l: l[0]) # Ordena da esquerda para a direita (A -> E)
        
        votos = []
        for (lx, ly, lw, lh, l_cnt) in linha:
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [l_cnt], -1, 255, -1)
            votos.append(cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask)))
        
        letras = ['A', 'B', 'C', 'D', 'E']
        escolha = letras[np.argmax(votos)] if max(votos) > 40 else "?"
        questoes_lidas.append({'x': linha[0][0], 'y': linha[0][1], 'res': escolha})

    # 4. ORDENAÇÃO POR COLUNAS (3 colunas por andar)
    # Dividimos a folha em 4 "andares" horizontais baseados no Y
    questoes_lidas.sort(key=lambda q: (q['y'] // 350, q['x']))

    # 5. DISTRIBUIÇÃO NAS MATÉRIAS (Garante o "pulo" correto de questões)
    resultado_final = {}
    ponteiro = 0
    for nome, qtd in estrutura_prova:
        resultado_final[nome] = []
        for _ in range(qtd):
            if ponteiro < len(questoes_lidas):
                resultado_final[nome].append(questoes_lidas[ponteiro]['res'])
                ponteiro += 1
            else:
                resultado_final[nome].append("?")
                
    return resultado_final
