import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def detectar_respostas_por_pixel(image_file, num_questoes):
    # Converte para OpenCV
    filestr = image_file.read()
    nparr = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {}

    # Pré-processamento: Tons de cinza e inversão (o que é marcado fica branco)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    respostas = {}
    alternativas = ['A', 'B', 'C', 'D', 'E']
    
    # Lógica de Fatiamento:
    # Para o seu gabarito, dividimos a altura total pelo número de questões
    h, w = thresh.shape
    fatia_h = h // num_questoes
    coluna_w = w // 5

    for i in range(num_questoes):
        y_topo = i * fatia_h
        y_base = (i + 1) * fatia_h
        
        counts = []
        for j in range(5):
            x_esq = j * coluna_w
            x_dir = (j + 1) * coluna_w
            
            # Corta o quadradinho da alternativa
            roi = thresh[y_topo:y_base, x_esq:x_dir]
            # Conta pixels brancos (marcação)
            counts.append(cv2.countNonZero(roi))
        
        # A alternativa com MAIS pixels marcados vence
        vencedor = np.argmax(counts)
        # Se houver uma marcação mínima, registra a letra. Se não, "?"
        respostas[str(i+1)] = alternativas[vencedor] if counts[vencedor] > 100 else "?"

    return respostas

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/extrair-gabarito", methods=["POST"])
def api_extrair_gabarito():
    try:
        file = request.files.get("imagem")
        num_q = int(request.form.get("num_questoes", 10))
        if not file:
            return jsonify({"erro": "Imagem não enviada"}), 400
        
        # Agora chama a função que REALMENTE analisa os pixels
        gabarito = detectar_respostas_por_pixel(file, num_q)
        return jsonify({"gabarito": gabarito})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/api/corrigir", methods=["POST"])
def api_corrigir():
    try:
        file = request.files.get("imagem")
        gabarito_str = request.form.get("gabarito")
        nome_aluno = request.form.get("nome_aluno", "Aluno")

        if not file or not gabarito_str:
            return jsonify({"erro": "Dados incompletos"}), 400

        gabarito = json.loads(gabarito_str)
        respostas_aluno = detectar_respostas_por_pixel(file, len(gabarito))

        acertos = 0
        detalhes = {}
        for q, resp_correta in gabarito.items():
            resp_aluno = respostas_aluno.get(q, "?")
            acertou = str(resp_aluno).upper() == str(resp_correta).upper()
            if acertou: 
                acertos += 1
            detalhes[q] = {"gabarito": resp_correta, "resposta": resp_aluno, "acertou": acertou}

        nota = round((acertos / len(gabarito)) * 10, 2) if gabarito else 0
        return jsonify({"nome_aluno": nome_aluno, "nota": nota, "acertos": acertos, "total": len(gabarito), "detalhes": detalhes})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
