import subprocess, sys, os
from flask import Flask, render_template, jsonify

app = Flask(__name__)
BASE = os.path.dirname(__file__)

ADAPTERS = [
    {
        "id": "accounting",
        "name": "Accounting Period Close",
        "desc": "Books off by $1,200 at Q3. Baru finds the 2 wrong entries and fixes them.",
        "file": "demo_accounting.py",
        "runnable": True,
        "tags": ["finance", "demo"],
    },
    {
        "id": "staff",
        "name": "Staff Rotation Cycle",
        "desc": "12 nurses, 12 slots. After a rough quarter the rotation is off by 2 slots. Baru fixes it in 1 swap.",
        "file": "demo_staff_rotation.py",
        "runnable": True,
        "tags": ["scheduling", "demo"],
    },
    {
        "id": "robot",
        "name": "Warehouse Robot Path",
        "desc": "Robot stranded 3.6m from dock after a bad route. Baru corrects 2 moves and gets it home.",
        "file": "demo_warehouse_robot.py",
        "runnable": True,
        "tags": ["robotics", "demo"],
    },
    {
        "id": "base",
        "name": "BaruAdapter Base Class",
        "desc": "The easy way to build any Baru adapter. Answer 4 questions, Baru does the rest. Includes Counter, Clock, Toggle, Steps built-ins.",
        "file": "BaruAdapter.py",
        "runnable": True,
        "tags": ["core", "toolkit"],
    },
    {
        "id": "catalog",
        "name": "Catalog OCR Extraction",
        "desc": "Fixes failed OCR scans in a grid-based catalog. Blurry text? Bad crop? Unknown model? Baru picks the right correction move.",
        "file": "G2_CatalogExtractionAdapter.py",
        "runnable": False,
        "tags": ["vision", "blueprint"],
    },
    {
        "id": "harmonic",
        "name": "Harmonic Cadence (Music)",
        "desc": "Resolves any chord back to the tonic using classical/jazz voice-leading rules. Circle of fifths, tritone subs, stepwise bass.",
        "file": "G3_HarmonicCadenceAdapter.py",
        "runnable": False,
        "tags": ["music", "blueprint"],
    },
    {
        "id": "g1base",
        "name": "Baru Base Interface (Gemini)",
        "desc": "Gemini's take on the Baru interface — 4 abstract methods defining any adapter domain.",
        "file": "G1_BaruAdapter_base.py",
        "runnable": False,
        "tags": ["core", "blueprint"],
    },
]


@app.route("/")
def index():
    return render_template("index.html", adapters=ADAPTERS)


@app.route("/run/<adapter_id>")
def run_adapter(adapter_id):
    adapter = next((a for a in ADAPTERS if a["id"] == adapter_id), None)
    if not adapter:
        return jsonify({"error": "Adapter not found"}), 404
    if not adapter["runnable"]:
        return jsonify({"error": "This is a blueprint — no runnable demo yet."}), 400

    script = os.path.join(BASE, adapter["file"])
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, timeout=15, cwd=BASE
    )
    output = result.stdout or result.stderr
    return jsonify({"output": output, "ok": result.returncode == 0})


@app.route("/source/<adapter_id>")
def source(adapter_id):
    adapter = next((a for a in ADAPTERS if a["id"] == adapter_id), None)
    if not adapter:
        return jsonify({"error": "Not found"}), 404
    path = os.path.join(BASE, adapter["file"])
    with open(path) as f:
        return jsonify({"source": f.read(), "file": adapter["file"]})


if __name__ == "__main__":
    app.run(debug=True, port=5050)
