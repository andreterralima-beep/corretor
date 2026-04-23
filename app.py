import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify, render_template

# Inicialização obrigatória para o Gunicorn encontrar o 'app'
app = Flask(__name__)

def detectar_por_blocos(image_file, num_questoes_total):
    # Converte imagem vinda do formulário para formato OpenCV
    filestr = image_file.read()
    nparr = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {str(i): "?" for i in range(1, num_questoes_total + 1)}

    # Pré-processamento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 1. Localizar os grandes blocos (matérias)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocos = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 100 and h > 80: # Filtra retângulos das disciplinas
            blocos.append((x, y, w, h))

    # Ordena blocos: cima para baixo, esquerda para direita
    blocos.sort(key=lambda b: (b[1] // 50, b[0]))

    respostas = {}
    q_idx = 1
    alternativas = ['A', 'B', 'C', 'D', 'E']

    for (bx, by, bw, bh) in blocos:
        roi = thresh[by:by+bh, bx:bx+bw]
        # Localiza os quadradinhos de cada questão dentro do bloco
        c_internos, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        sqs = []
        for ci in c_internos:
            ix, iy, iw, ih = cv2.boundingRect(ci)
            ar = iw / float(ih)
            if 15 < iw < 45 and 0.7 < ar < 1.3:
                sqs.append((ix, iy, iw, ih, ci))
        
        # Ordena por linha
        sqs.sort(key=lambda s: s[1])

        # Agrupa de 5 em 5 (as 5 alternativas de uma questão)
        for i in range(0, len(sqs), 5):
            if q_idx > num_questoes_total: break
            
            linha = sqs[i:i+5]
            if len(linha) < 5: continue
            linha.sort(key=lambda s: s[0]) # Ordem A, B, C, D, E

            votos = []
            for (lx, ly, lw, lh, l_cnt) in linha:
                mask = np.zeros(roi.shape, dtype="uint8")
                cv2.drawContours(mask, [l_cnt], -1, 255, -1)
                mask = cv2.bitwise_and(roi, roi, mask=mask)
                votos.append(cv2.countNonZero(mask))
            
            respostas[str(q_idx)] = alternativas[np.argmax(votos)] if max(votos) > 40 else "?"
            q_idx += 1

    # Preenche o restante se o OpenCV pular algo
    for i in range(1, num_questoes_total + 1):
        if str(i) not in respostas: respostas[str(i)] = "?"
            
    return respostas

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/extrair-gabarito", methods=["POST"])
def api_extrair_gabarito():
    try:
        file = request.files.get("imagem")
        num_q = int(request.form.get("num_questoes", 50))
        if not file: return jsonify({"erro": "Sem imagem"}), 400
        
        res = detectar_por_blocos(file, num_q)
        return jsonify({"gabarito": res})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/api/corrigir", methods=["POST"])
def api_corrigir():
    try:
        file = request.files.get("imagem")
        gabarito = json.loads(request.form.get("gabarito"))
        nome = request.form.get("nome_aluno", "Aluno")

        res_aluno = detectar_por_blocos(file, len(gabarito))

        acertos = 0
        detalhes = {}
        for q, resp_correta in gabarito.items():
            resp_aluno = res_aluno.get(q, "?")
            acertou = str(resp_aluno) == str(resp_correta)
            if acertou: acertos += 1
            detalhes[q] = {"gabarito": resp_correta, "resposta": resp_aluno, "acertou": acertou}

        nota = round((acertos / len(gabarito)) * 10, 2)
        return jsonify({"nome_aluno": nome, "nota": nota, "acertos": acertos, "total": len(gabarito), "detalhes": detalhes})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
