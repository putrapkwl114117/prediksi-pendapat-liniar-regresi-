from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('models/model_furniture.pkl', 'rb') as file:
    model = pickle.load(file)

# Fitur untuk prediksi
features = ['Harga', 'Biaya', 'Produk Terjual', 'Keuntungan', 'Stok Barang', 'Diskon', 'Hari Pengiriman']

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form
        harga = float(request.form['harga'])
        biaya = float(request.form['biaya'])
        produk_terjual = float(request.form['produk_terjual'])
        keuntungan = float(request.form['keuntungan'])
        stok_barang = float(request.form['stok_barang'])
        diskon = float(request.form['diskon'])
        hari_pengiriman = float(request.form['hari_pengiriman'])

        # Gabungkan semua input ke dalam array numpy
        input_data = np.array([[harga, biaya, produk_terjual, keuntungan, stok_barang, diskon, hari_pengiriman]])

        # Prediksi menggunakan model
        prediction = model.predict(input_data)

        # Ambil koefisien model jika model adalah regresi linier
        if hasattr(model, 'coef_'):
            if isinstance(model.coef_, np.ndarray):
                if model.coef_.ndim == 2:  # Jika koefisien 2D (multiple output)
                    coefficients = model.coef_[0]
                else:  # Jika koefisien 1D
                    coefficients = model.coef_
            else:
                coefficients = [model.coef_]
        else:
            coefficients = ["Model tidak memiliki koefisien"]

        # Pastikan bahwa coefficients adalah iterable
        if isinstance(coefficients, np.ndarray) and coefficients.ndim == 1:
            coefficients = coefficients.tolist()

        # Menggabungkan fitur dan koefisien
        feature_coeffs = zip(features, coefficients)

        # Menyesuaikan penjelasan berdasarkan koefisien
        explanations = {}
        for feature, coeff in zip(features, coefficients):
            if coeff > 0:
                if coeff > 1000000:  # Koefisien besar, pengaruh signifikan
                    explanations[feature] = f'{feature} memiliki pengaruh besar terhadap pendapatan dan perlu dioptimalkan.'
                else:
                    explanations[feature] = f'{feature} memiliki pengaruh positif terhadap pendapatan. Meningkatkan {feature} akan meningkatkan pendapatan.'
            elif coeff < 0:
                if coeff < -1000000:  # Koefisien negatif besar, pengaruh besar
                    explanations[feature] = f'{feature} memiliki pengaruh besar dan negatif terhadap pendapatan, sehingga harus diminimalkan.'
                else:
                    explanations[feature] = f'{feature} memiliki pengaruh negatif terhadap pendapatan. Meningkatkan {feature} akan menurunkan pendapatan.'
            else:
                explanations[feature] = f'{feature} tidak memiliki pengaruh signifikan terhadap pendapatan.'

        # Kesimpulan berdasarkan pengaruh terbesar
        max_positive_feature = max(zip(features, coefficients), key=lambda x: x[1] if x[1] > 0 else -float('inf'))
        max_negative_feature = min(zip(features, coefficients), key=lambda x: x[1] if x[1] < 0 else float('inf'))

        # Kesimpulan akhir
        conclusion = ""
        if max_positive_feature[1] > 0:
            conclusion += f"Faktor yang paling berpengaruh positif terhadap pendapatan adalah {max_positive_feature[0]}. Ini harus dioptimalkan untuk meningkatkan pendapatan. "
        if max_negative_feature[1] < 0:
            conclusion += f"Faktor yang paling berpengaruh negatif terhadap pendapatan adalah {max_negative_feature[0]}. Ini harus diminimalkan untuk menghindari penurunan pendapatan."

        return render_template('result.html', prediction=prediction[0], feature_coeffs=feature_coeffs, explanations=explanations, conclusion=conclusion)

    except Exception as e:
        return f"Terjadi kesalahan: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
