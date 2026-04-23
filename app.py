import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def detectar_pelo_formato(image_file):
    # Converte imagem para OpenCV
    filestr = image_file.read()
    nparr = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return {}

    # 1. Transformar em Preto e Branco de alto contraste
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 2. Encontrar todos os "quadradinhos" das alternativas
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    opcoes_detectadas = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # Filtra apenas objetos que tenham tamanho e formato de uma "bolinha/quadrado" de resposta
        if w >= 15 and h >= 15 and 0.8 <= ar <= 1.2:
            # Calcula a intensidade de preenchimento
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            opcoes_detectadas.append({'x': x, 'y': y, 'total': total})

    # 3. Ordenar e Agrupar
    # Como o gabarito é complexo, vamos ordenar por posição Y (linhas) e X (colunas)
    opcoes_detectadas.sort(key=lambda b: (b['y'], b['x']))

    respostas = {}
    alternativas = ['A', 'B', 'C', 'D', 'E']
    
    # Agrupamos de 5 em 5 (cada questão)
    for i in range(0, len(opcoes_detectadas) // 5):
        questao = opcoes_detectadas[i*5 : (i+1)*5]
        questao.sort(key=lambda b: b['x']) # Garante ordem A, B, C, D, E
        
        # A que tiver mais pixels brancos (preenchimento) é a marcada
        votos = [o['total'] for o in questao]
        ganhador = np.argmax(votos)
        
        # Só conta se o preenchimento for significativo (evita ler o quadrado vazio)
        if votos[ganhador] > 50: 
            respostas[str(i + 1)] = alternativas[ganhador]
        else:
            respostas[str(i + 1)] = "?"

    return respostas

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/extrair-gabarito", methods=["POST"])
def api_extrair_gabarito():
    try:
        file = request.files.get("imagem")
        if not file: return jsonify({"erro": "Imagem não enviada"}), 400
        gabarito = detectar_pelo_formato(file)
        return jsonify({"gabarito": gabarito})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/api/corrigir", methods=["POST"])
def api_corrigir():
    try:
        file = request.files.get("imagem")
        gabarito = json.loads(request.form.get("gabarito"))
        nome_aluno = request.form.get("nome_aluno", "Aluno")

        respostas_aluno = detectar_pelo_formato(file)

        acertos = 0
        detalhes = {}
        for q, resp_correta in gabarito.items():
            resp_aluno = respostas_aluno.get(q, "?")
            acertou = str(resp_aluno) == str(resp_correta)
            if acertou: acertos += 1
            detalhes[q] = {"gabarito": resp_correta, "resposta": resp_aluno, "acertou": acertou}

        nota = round((acertos / len(gabarito)) * 10, 2) if gabarito else 0
        return jsonify({"nome_aluno": nome_aluno, "nota": nota, "acertos": acertos, "total": len(gabarito), "detalhes": detalhes})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
