import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify, render_template

# Gunicorn precisa desta variável 'app' no nível zero do arquivo
app = Flask(__name__)

def processar_gabarito_jales(image_file, num_total):
    # Lê a imagem do buffer
    filestr = image_file.read()
    nparr = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {str(i): "?" for i in range(1, num_total + 1)}

    # 1. PRÉ-PROCESSAMENTO: Focar em contornos de caixas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold adaptativo para fotos com iluminação irregular
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 2. DETECTAR BLOCOS DE MATÉRIAS
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocos = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Filtra retângulos compatíveis com os blocos de matérias
        if w > 150 and h > 80:
            blocos.append((x, y, w, h))

    # ORDENAÇÃO POR COLUNAS E LINHAS (Crucial para o layout de 3 colunas)
    # Agrupa blocos em "faixas" de 100px de altura para ler da esquerda para a direita
    blocos.sort(key=lambda b: (b[1] // 100, b[0]))

    respostas = {}
    q_global = 1
    letras = ['A', 'B', 'C', 'D', 'E']

    for (bx, by, bw, bh) in blocos:
        roi = thresh[by:by+bh, bx:bx+bw]
        # Encontra os quadradinhos das alternativas dentro do bloco
        c_internos, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        sqs = []
        for ci in c_internos:
            ix, iy, iw, ih = cv2.boundingRect(ci)
            ar = iw / float(ih)
            # Filtra apenas o tamanho de uma "bolinha" de gabarito
            if 15 < iw < 45 and 0.7 < ar < 1.3:
                sqs.append((ix, iy, iw, ih, ci))
        
        # Ordena as alternativas do bloco de cima para baixo
        sqs.sort(key=lambda s: s[1])

        # Processa cada linha de 5 alternativas
        for i in range(0, len(sqs), 5):
            if q_global > num_total: break
            
            linha = sqs[i:i+5]
            if len(linha) < 5: continue
            linha.sort(key=lambda s: s[0]) # Ordem A, B, C, D, E

            votos = []
            for (lx, ly, lw, lh, l_cnt) in linha:
                mask = np.zeros(roi.shape, dtype="uint8")
                cv2.drawContours(mask, [l_cnt], -1, 255, -1)
                mask = cv2.bitwise_and(roi, roi, mask=mask)
                votos.append(cv2.countNonZero(mask))
            
            # Se o maior preenchimento for significativo, registra a letra
            maior_voto = np.argmax(votos)
            if votos[maior_voto] > 45: # Sensibilidade do preenchimento
                respostas[str(q_global)] = letras[maior_voto]
            else:
                respostas[str(q_global)] = "?"
            q_global += 1

    # Preenchimento de segurança para garantir que a tabela no HTML não quebre
    for i in range(1, num_total + 1):
        if str(i) not in respostas: respostas[str(i)] = "?"
            
    return respostas

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/extrair-gabarito", methods=["POST"])
def api_extrair():
    try:
        f = request.files.get('imagem')
        n = int(request.form.get('num_questoes', 50))
        res = processar_gabarito_jales(f, n)
        return jsonify({"gabarito": res})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/api/corrigir", methods=["POST"])
def api_corrigir():
    try:
        f = request.files.get('imagem')
        gab = json.loads(request.form.get('gabarito'))
        nome = request.form.get('nome_aluno', 'Estudante')
        
        # Detecta as respostas na folha do aluno
        res_aluno = processar_gabarito_jales(f, len(gab))
        
        acertos = 0
        for q, correta in gab.items():
            if str(res_aluno.get(q)) == str(correta):
                acertos += 1
        
        nota = round((acertos / len(gab)) * 10, 2)
        return jsonify({"nome": nome, "nota": nota, "acertos": acertos, "total": len(gab)})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
