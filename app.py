import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def detectar_respostas_robusto(image_file, num_questoes):
    respostas = {}
    # Inicializa todas as questões como "?" para garantir que a tabela apareça
    for i in range(1, num_questoes + 1):
        respostas[str(i)] = "?"

    try:
        filestr = image_file.read()
        nparr = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return respostas

        # Processamento de imagem
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)

        # Localiza contornos (quadradinhos das respostas)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidatos = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if w >= 15 and h >= 15 and 0.7 <= ar <= 1.3:
                candidatos.append({'x': x, 'y': y, 'c': c})

        # Se encontrou marcações, tenta organizar
        if candidatos:
            candidatos.sort(key=lambda b: (b['y'], b['x']))
            alternativas = ['A', 'B', 'C', 'D', 'E']
            
            # Tenta preencher as questões detetadas
            for i in range(min(len(candidatos) // 5, num_questoes)):
                questao_bloc = candidatos[i*5 : (i+1)*5]
                questao_bloc.sort(key=lambda b: b['x'])
                
                intensidades = []
                for opt in questao_bloc:
                    mask = np.zeros(thresh.shape, dtype="uint8")
                    cv2.drawContours(mask, [opt['c']], -1, 255, -1)
                    mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                    intensidades.append(cv2.countNonZero(mask))
                
                vencedor = np.argmax(intensidades)
                if intensidades[vencedor] > 40:
                    respostas[str(i + 1)] = alternativas[vencedor]

    except Exception:
        pass # Se der erro na imagem, retorna a lista com "?"

    return respostas

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/extrair-gabarito", methods=["POST"])
def api_extrair_gabarito():
    try:
        file = request.files.get("imagem")
        num_q = int(request.form.get("num_questoes", 10))
        # Agora a função garante que sempre retorna o número correto de questões
        gabarito = detectar_respostas_robusto(file, num_q)
        return jsonify({"gabarito": gabarito})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/api/corrigir", methods=["POST"])
def api_corrigir():
    try:
        file = request.files.get("imagem")
        gabarito = json.loads(request.form.get("gabarito"))
        nome_aluno = request.form.get("nome_aluno", "Aluno")

        respostas_aluno = detectar_respostas_robusto(file, len(gabarito))

        acertos = 0
        detalhes = {}
        for q, resp_correta in gabarito.items():
            resp_aluno = respostas_aluno.get(q, "?")
            acertou = str(resp_aluno) == str(resp_correta)
            if acertou: acertos += 1
            detalhes[q] = {"gabarito": resp_correta, "resposta": resp_aluno, "acertou": acertou}

        nota = round((acertos / len(gabarito)) * 10, 2) if gabarito else 0
        return jsonify({
            "nome_aluno": nome_aluno, 
            "nota": nota, 
            "acertos": acertos, 
            "total": len(gabarito), 
            "detalhes": detalhes
        })
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
