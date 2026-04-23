import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def detectar_respostas(image_file):
    # Estrutura real da folha Jales Machado
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

    # Converte para HSV para isolar as cores
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- MÁSCARA 1: AZUL (Canetas esferográficas comuns) ---
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # --- MÁSCARA 2: PRETO (Canetas pretas ou azul muito escuro) ---
    # Focamos em tons de baixa luminosidade (V baixo)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 75]) 
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    
    # COMBINAÇÃO: O que for azul OU preto vira alvo
    mask_final = cv2.bitwise_or(mask_blue, mask_black)
    
    # Limpeza de ruído (pontinhos isolados)
    kernel = np.ones((3,3), np.uint8)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)

    # 1. Detectar os quadradinhos (templates) para servir de guia
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_boxes = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    cnts, _ = cv2.findContours(thresh_boxes.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 18 < w < 50 and 0.7 < (w/h) < 1.3:
            boxes.append((x, y, w, h, c))

    boxes.sort(key=lambda b: b[1]) # Ordena por altura

    # 2. Analisar preenchimento dentro de cada box
    questoes_lidas = []
    for i in range(0, len(boxes), 5):
        linha = boxes[i:i+5]
        if len(linha) < 5: continue
        linha.sort(key=lambda l: l[0]) # Ordena A, B, C, D, E
        
        votos = []
        for (bx, by, bw, bh, b_cnt) in linha:
            # Recorta a máscara de cor apenas no espaço desse quadradinho
            roi_mask = mask_final[by:by+bh, bx:bx+bw]
            votos.append(cv2.countNonZero(roi_mask))
        
        letras = ['A', 'B', 'C', 'D', 'E']
        # Se o quadradinho com mais "tinta" (azul ou preta) passar de 30 pixels, marca a letra
        escolha = letras[np.argmax(votos)] if max(votos) > 30 else "?"
        questoes_lidas.append({'x': linha[0][0], 'y': linha[0][1], 'res': escolha})

    # 3. Organização por colunas e andares (Layout 3 colunas)
    questoes_lidas.sort(key=lambda q: (q['y'] // 350, q['x']))

    resultado = {}
    ponteiro = 0
    for nome, qtd in estrutura_prova:
        resultado[nome] = []
        for _ in range(qtd):
            if ponteiro < len(questoes_lidas):
                resultado[nome].append(questoes_lidas[ponteiro]['res'])
                ponteiro += 1
            else:
                resultado[nome].append("?")
                
    return resultado

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/extrair-gabarito", methods=["POST"])
def api_extrair():
    try:
        f = request.files.get('imagem')
        return jsonify({"gabarito": detectar_respostas(f)})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/api/corrigir", methods=["POST"])
def api_corrigir():
    try:
        f = request.files.get('imagem')
        gab_mestre = json.loads(request.form.get('gabarito'))
        nome = request.form.get('nome_aluno', 'Estudante')
        res_aluno = detectar_respostas(f)

        acertos, total = 0, 0
        for mat, q_mestre in gab_mestre.items():
            q_aluno = res_aluno.get(mat, [])
            for idx, resp_mestra in enumerate(q_mestre):
                total += 1
                if idx < len(q_aluno) and str(q_aluno[idx]) == str(resp_mestra):
                    acertos += 1
        
        nota = round((acertos / total) * 10, 2) if total > 0 else 0
        return jsonify({"nome": nome, "nota": nota, "acertos": acertos, "total": total})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
