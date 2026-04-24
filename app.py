import os, cv2, numpy as np, json, base64
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

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
    pts = pts.reshape(4,2)
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
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)

    cnts,_ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            pts = ordenar_pontos(approx)

            (tl, tr, br, bl) = pts

            largura = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            altura  = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

            destino = np.array([
                [0,0],[largura-1,0],[largura-1,altura-1],[0,altura-1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(pts, destino)
            return cv2.warpPerspective(img, M, (largura, altura))

    return img


# ==============================
# DETECÇÃO
# ==============================
def detectar_respostas(image_file, config=None):

    raw = image_file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        return {}, ""

    cfg = {
        "thresh_block": 15,
        "thresh_C": 5,
        "fill_min": 0.25,
        "blur": 5
    }

    if config:
        cfg.update(config)

    img = corrigir_perspectiva(img)
    img = cv2.resize(img, (1200, 1700))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, cfg["blur"])
    gray = cv2.equalizeHist(gray)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        cfg["thresh_block"],
        cfg["thresh_C"]
    )

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    h, w = thresh.shape

    colunas = [int(w*0.17), int(w*0.50), int(w*0.83)]
    y_inicial = int(h*0.24)
    passo_y   = int(h*0.045)
    passo_x   = int(w*0.045)

    resultado = {}

    for i_col, col_x in enumerate(colunas):
        estrutura_col = ESTRUTURA[f"col{i_col+1}"]
        y = y_inicial

        for materia, qtd in estrutura_col:
            respostas = []

            for _ in range(qtd):
                marcacoes = []

                for alt in range(5):
                    x = col_x + alt * passo_x
                    roi = thresh[y:y+35, x:x+35]

                    if roi.shape[0] == 0:
                        marcacoes.append(0)
                        continue

                    area = cv2.countNonZero(roi)
                    total = roi.shape[0] * roi.shape[1]

                    marcacoes.append(area / total)

                idx = np.argmax(marcacoes)

                if marcacoes[idx] > cfg["fill_min"]:
                    respostas.append(LETRAS[idx])
                else:
                    respostas.append("?")

                y += passo_y

            resultado[materia] = respostas

    debug = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    ds = cv2.resize(debug, (600, 850))
    _, buf = cv2.imencode(".jpg", ds)

    return resultado, base64.b64encode(buf).decode()


# ==============================
# ROTAS
# ==============================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/extrair-gabarito", methods=["POST"])
def api_extrair():
    try:
        f = request.files.get("imagem")

        config = {
            "thresh_block": int(request.form.get("thresh_block", 15)),
            "thresh_C": int(request.form.get("thresh_C", 5)),
            "fill_min": float(request.form.get("fill_min", 0.25)),
            "blur": int(request.form.get("blur", 5))
        }

        gabarito, debug_b64 = detectar_respostas(f, config)

        return jsonify({"gabarito":gabarito,"debug_img":debug_b64})

    except Exception as e:
        return jsonify({"erro":str(e)}),500


if __name__ == "__main__":
    app.run(debug=True)
