from flask import Flask, render_template, request, redirect, session
import pandas as pd
import torch, pickle
from transformers import BertTokenizer, BertModel

app = Flask(__name__)
app.secret_key = "rrd-secret"

# ================= LOAD MODEL =================
print("Load BERT...")
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
bert = BertModel.from_pretrained("indobenchmark/indobert-base-p1")

print("Load Random Forest...")
rf = pickle.load(open("model_rf.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = bert(**inputs)
    return out.last_hidden_state[:,0,:].numpy()

# =============== MAHASISWA ====================
@app.route('/')
def mahasiswa_page():
    return render_template("mahasiswa.html")

@app.route('/predict', methods=['POST'])
def predict():
    judul = request.form['judul']
    emb = get_embedding(judul)
    pred = rf.predict(emb)
    dosen = le.inverse_transform(pred)[0]

    # Ambil keahlian dosen dari dosen.csv
    df_dosen = pd.read_csv("dosen.csv")
    row = df_dosen[df_dosen['nama_dosen'] == dosen].iloc[0]
    keahlian = row['keahlian']

    # TIDAK disimpan ke riwayat
    return render_template("hasil.html", judul=judul, dosen=dosen, keahlian=keahlian)

# =============== LOGIN ADMIN ==================
@app.route('/admin')
def admin_login():
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login():
    if request.form['user'] == "admin" and request.form['pass'] == "tika2004":
        session['admin'] = True
        return redirect("/dashboard")
    return redirect("/admin")

@app.route('/logout-confirm')
def logout_confirm():
    if not session.get('admin'):
        return redirect("/admin")
    return render_template("logout_confirm.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect("/admin")

# =============== DASHBOARD ====================
@app.route('/dashboard')
def dashboard():
    if not session.get('admin'):
        return redirect("/admin")
    return render_template("dashboard.html")

# ============== KELOLA DOSEN ==================
@app.route('/kelola-dosen')
def kelola_dosen():
    if not session.get('admin'):
        return redirect("/admin")
    df = pd.read_csv("dosen.csv")
    return render_template("kelola_dosen.html", data=df.to_dict('records'))

@app.route('/tambah-dosen', methods=['GET','POST'])
def tambah_dosen():
    if not session.get('admin'):
        return redirect("/admin")

    if request.method == "POST":
        nama = request.form['nama']
        keahlian = request.form['keahlian']

        df = pd.read_csv("dosen.csv")
        df.loc[len(df)] = {"nama_dosen": nama, "keahlian": keahlian}
        df.to_csv("dosen.csv", index=False)
        return redirect("/kelola-dosen")

    return render_template("tambah_dosen.html")

@app.route('/edit-dosen/<nama>', methods=['GET','POST'])
def edit_dosen(nama):
    if not session.get('admin'):
        return redirect("/admin")

    df = pd.read_csv("dosen.csv")

    if request.method == "POST":
        new_nama = request.form['nama']
        new_keahlian = request.form['keahlian']

        df.loc[df['nama_dosen'] == nama, 'nama_dosen'] = new_nama
        df.loc[df['nama_dosen'] == new_nama, 'keahlian'] = new_keahlian
        df.to_csv("dosen.csv", index=False)
        return redirect("/kelola-dosen")

    row = df[df['nama_dosen'] == nama].iloc[0]
    return render_template("edit_dosen.html", data=row.to_dict())

@app.route('/hapus-dosen/<nama>')
def hapus_dosen(nama):
    if not session.get('admin'):
        return redirect("/admin")
    df = pd.read_csv("dosen.csv")
    df = df[df['nama_dosen'] != nama]
    df.to_csv("dosen.csv", index=False)
    return redirect("/kelola-dosen")

# ============== RIWAYAT REKOMENDASI ===========
@app.route('/riwayat')
def riwayat():
    if not session.get('admin'):
        return redirect("/admin")
    df = pd.read_csv("riwayat.csv")
    return render_template("riwayat.html", data=df.to_dict('records'))

@app.route('/tambah-riwayat', methods=['GET','POST'])
def tambah_riwayat():
    if not session.get('admin'):
        return redirect("/admin")

    if request.method == "POST":
        judul = request.form['judul']
        dosen = request.form['dosen']

        df = pd.read_csv("riwayat.csv")
        df.loc[len(df)] = {"judul": judul, "dosen": dosen}
        df.to_csv("riwayat.csv", index=False)
        return redirect("/riwayat")

    return render_template("tambah_riwayat.html")

@app.route('/edit-riwayat/<int:index>', methods=['GET','POST'])
def edit_riwayat(index):
    if not session.get('admin'):
        return redirect("/admin")

    df = pd.read_csv("riwayat.csv")

    if request.method == "POST":
        df.at[index, 'judul'] = request.form['judul']
        df.at[index, 'dosen'] = request.form['dosen']
        df.to_csv("riwayat.csv", index=False)
        return redirect("/riwayat")

    row = df.iloc[index]
    return render_template("edit_riwayat.html", data=row.to_dict(), index=index)

@app.route('/hapus-riwayat/<int:index>')
def hapus_riwayat(index):
    if not session.get('admin'):
        return redirect("/admin")

    df = pd.read_csv("riwayat.csv")
    df = df.drop(index)
    df.to_csv("riwayat.csv", index=False)
    return redirect("/riwayat")

# ================= RUN APP ====================
if __name__ == '__main__':
    app.run(debug=True)
