import os, cv2, numpy as np, json, base64
from flask import Flask, request, jsonify, render_template, session

app = Flask(__name__)
app.secret_key = "corretor-secret-2026"

ESTRUTURA = {
    "col1": [("Língua Portuguesa",10),("Química",4),("Geografia",4),("Arte",2)],
    "col2": [("Matemática",10),("Física",4),("Língua Inglesa",2),("Sociologia",2)],
    "col3": [("Biologia",4),("História",4),("Educação Física",2),("Filosofia",2)],
}

LETRAS = ["A","B","C","D","E"]

# ==============================
# PERSPECTIVA
# ==============================
def ordenar_pontos(pts):
    pts = pts.reshape(4, 2)
    soma = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(soma)],
        pts[np.argmin(diff)],
        pts[np.argmax(soma)],
        pts[np.argmax(diff)]
    ], dtype="float32")

def corrigir_perspectiva(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 30, 120)
    kernel = np.ones((3, 3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)

    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            pts = ordenar_pontos(approx)
            (tl, tr, br, bl) = pts
            largura = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            altura = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
            if largura < 100 or altura < 100:
                continue
            destino = np.array([
                [0, 0], [largura-1, 0], [largura-1, altura-1], [0, altura-1]
            ], dtype="float32")
            M = cv2.getPerspectiveTransform(pts, destino)
            return cv2.warpPerspective(img, M, (largura, altura))
    return img

# ==============================
# DETECÇÃO — CANETA AZUL E PRETA
# ==============================
def criar_mascara_marcacao(img_bgr, cfg):
    """Detecta marcações de caneta azul OU preta."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_k = cfg["blur"] if cfg["blur"] % 2 == 1 else cfg["blur"] + 1
    gray = cv2.medianBlur(gray, blur_k)

    # Máscara 1: caneta preta via threshold adaptativo
    eq = cv2.equalizeHist(gray)
    thresh_preta = cv2.adaptiveThreshold(
        eq, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        cfg["thresh_block"],
        cfg["thresh_C"]
    )

    # Máscara 2: caneta azul via canal HSV (detecção de cor azul)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 30])
    upper_blue = np.array([140, 255, 255])
    thresh_azul = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combina as duas máscaras
    combinado = cv2.bitwise_or(thresh_preta, thresh_azul)

    # Limpeza morfológica
    kernel = np.ones((3, 3), np.uint8)
    combinado = cv2.morphologyEx(combinado, cv2.MORPH_OPEN, kernel)
    combinado = cv2.dilate(combinado, kernel, iterations=1)

    return combinado, thresh_preta, thresh_azul

def detectar_respostas(img_bytes, config=None):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return {}, ""

    cfg = {
        "thresh_block": 15,
        "thresh_C": 5,
        "fill_min": 0.20,
        "blur": 5
    }
    if config:
        cfg.update(config)

    img = corrigir_perspectiva(img)
    img = cv2.resize(img, (1200, 1700))

    mascara, _, _ = criar_mascara_marcacao(img, cfg)

    h, w = mascara.shape
    colunas = [int(w*0.17), int(w*0.50), int(w*0.83)]
    y_inicial = int(h*0.24)
    passo_y   = int(h*0.045)
    passo_x   = int(w*0.045)

    resultado = {}
    debug = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)

    for i_col, col_x in enumerate(colunas):
        estrutura_col = ESTRUTURA[f"col{i_col+1}"]
        y = y_inicial

        for materia, qtd in estrutura_col:
            respostas = []

            for q in range(qtd):
                marcacoes = []

                for alt in range(5):
                    x = col_x + alt * passo_x
                    roi = mascara[y:y+35, x:x+35]

                    if roi.shape[0] == 0 or roi.shape[1] == 0 or roi.size == 0:
                        marcacoes.append(0.0)
                        continue

                    area = cv2.countNonZero(roi)
                    total = roi.shape[0] * roi.shape[1]
                    marcacoes.append(area / total if total > 0 else 0.0)

                    # Desenha retângulo de debug
                    cor = (0, 200, 0) if alt == np.argmax(marcacoes) else (60, 60, 60)
                    cv2.rectangle(debug, (x, y), (x+35, y+35), cor, 1)

                idx = int(np.argmax(marcacoes))
                letra = LETRAS[idx] if marcacoes[idx] > cfg["fill_min"] else "?"

                # Destaca a célula escolhida
                xm = col_x + idx * passo_x
                cor_dest = (0, 255, 0) if letra != "?" else (0, 0, 255)
                cv2.rectangle(debug, (xm, y), (xm+35, y+35), cor_dest, 2)

                respostas.append(letra)
                y += passo_y

            resultado[materia] = respostas

    ds = cv2.resize(debug, (600, 850))
    _, buf = cv2.imencode(".jpg", ds)
    debug_b64 = base64.b64encode(buf).decode()

    return resultado, debug_b64


# ==============================
# ROTAS
# ==============================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/extrair-gabarito", methods=["POST"])
def api_extrair_gabarito():
    try:
        f = request.files.get("imagem")
        if not f:
            return jsonify({"erro": "Imagem não enviada"}), 400

        config = {
            "thresh_block": int(request.form.get("thresh_block", 15)),
            "thresh_C":     int(request.form.get("thresh_C", 5)),
            "fill_min":     float(request.form.get("fill_min", 0.20)),
            "blur":         int(request.form.get("blur", 5))
        }

        img_bytes = f.read()
        gabarito, debug_b64 = detectar_respostas(img_bytes, config)

        # Salva gabarito na sessão
        session["gabarito"] = gabarito

        return jsonify({"gabarito": gabarito, "debug_img": debug_b64})

    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/api/salvar-gabarito", methods=["POST"])
def api_salvar_gabarito():
    """Salva gabarito editado manualmente pelo professor."""
    try:
        data = request.get_json()
        gabarito = data.get("gabarito")
        if not gabarito:
            return jsonify({"erro": "Gabarito vazio"}), 400
        session["gabarito"] = gabarito
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/api/corrigir-aluno", methods=["POST"])
def api_corrigir_aluno():
    try:
        f = request.files.get("imagem")
        if not f:
            return jsonify({"erro": "Imagem não enviada"}), 400

        config = {
            "thresh_block": int(request.form.get("thresh_block", 15)),
            "thresh_C":     int(request.form.get("thresh_C", 5)),
            "fill_min":     float(request.form.get("fill_min", 0.20)),
            "blur":         int(request.form.get("blur", 5))
        }

        img_bytes = f.read()
        respostas_aluno, debug_b64 = detectar_respostas(img_bytes, config)

        gabarito = session.get("gabarito")

        comparacao = {}
        total_q = 0
        total_acertos = 0

        for materia, respostas in respostas_aluno.items():
            gab = gabarito.get(materia, []) if gabarito else []
            detalhes = []

            for i, resp in enumerate(respostas):
                correto = gab[i] if i < len(gab) else "?"
                acerto = (resp == correto) and (resp != "?")
                detalhes.append({
                    "questao": i + 1,
                    "aluno": resp,
                    "gabarito": correto,
                    "acerto": acerto
                })
                total_q += 1
                if acerto:
                    total_acertos += 1

            comparacao[materia] = detalhes

        nota = round((total_acertos / total_q) * 10, 2) if total_q > 0 else 0

        return jsonify({
            "respostas": respostas_aluno,
            "comparacao": comparacao,
            "acertos": total_acertos,
            "total": total_q,
            "nota": nota,
            "debug_img": debug_b64,
            "tem_gabarito": gabarito is not None
        })

    except Exception as e:
        return jsonify({"erro": str(e)}), 500


@app.route("/api/gabarito-atual", methods=["GET"])
def api_gabarito_atual():
    gabarito = session.get("gabarito")
    return jsonify({"gabarito": gabarito})


if __name__ == "__main__":
    app.run(debug=True)
