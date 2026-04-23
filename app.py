import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify, render_template

# ESSA LINHA É O QUE RESOLVE O ERRO DO GUNICORN
app = Flask(__name__)

def detectar_por_materias_preciso(image_file):
    # ESTRUTURA EXATA DO SEU MODELO
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

    # 1. Captura quadradinhos de resposta
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quadradinhos = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 18 < w < 48 and 0.7 < (w/h) < 1.3:
            quadradinhos.append((x, y, w, h, c))

    quadradinhos.sort(key=lambda q: q[1])

    questoes_lidas = []
    for i in range(0, len(quadradinhos), 5):
        linha = quadradinhos[i:i+5]
        if len(linha) < 5: continue
        linha.sort(key=lambda l: l[0])
        
        votos = []
        for (lx, ly, lw, lh, l_cnt) in linha:
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [l_cnt], -1, 255, -1)
            votos.append(cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask)))
        
        letras = ['A', 'B', 'C', 'D', 'E']
        # Verifica se há preenchimento suficiente
        escolha = letras[np.argmax(votos)] if max(votos) > 40 else "?"
        questoes_lidas.append({'x': linha[0][0], 'y': linha[0][1], 'res': escolha})

    # Ordenação por colunas (3 por andar)
    questoes_lidas.sort(key=lambda q: (q['y'] // 350, q['x']))

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

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/extrair-gabarito", methods=["POST"])
def api_extrair():
    try:
        f = request.files.get('imagem')
        return jsonify({"gabarito": detectar_por_materias_preciso(f)})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

@app.route("/api/corrigir", methods=["POST"])
def api_corrigir():
    try:
        f = request.files.get('imagem')
        gab_mestre = json.loads(request.form.get('gabarito'))
        nome = request.form.get('nome_aluno', 'Estudante')
        res_aluno = detectar_por_materias_preciso(f)

        acertos, total = 0, 0
        for mat, q_mestre in gab_mestre.items():
            q_aluno = res_aluno.get(mat, [])
            for idx, resp in enumerate(q_mestre):
                total += 1
                if idx < len(q_aluno) and str(q_aluno[idx]) == str(resp):
                    acertos += 1
        
        nota = round((acertos / total) * 10, 2) if total > 0 else 0
        return jsonify({"nome": nome, "nota": nota, "acertos": acertos, "total": total})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
