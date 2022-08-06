# **Movie Recommendation System with Collaborative Filtering**

## **Project Overview**

Pertumbuhan pasar industri dari film luar negeri ke film dalam negeri semakin menjanjikan. Dilihat dari  jumlah penonton bioskop yang terus meningkat setiap tahunnya. Pada tahun 2018, jumlah penonton bioskop di Indonesia sendiri mencapai lebih dari 50 juta, dengan jumlah produk film luar dan dalam negeri mencapai hampir 200 judul  yang diputar di seluruh Indonesia [1].

Dari sekian banyak film yang diproduksi, sulit bagi calon penonton untuk memutuskan mana yang akan ditonton. Menemukan sebuah film tentu saja membutuhkan waktu, apalagi film tersebut telah menentukan bahwa itu  belum tentu diinginkan oleh calon penonton setelah menonton, sehingga mereka akan menghabiskan  lebih banyak waktu untuk itu. Menonton film di bioskop, platform penyedia layanan streaming, serta menyewa dan membeli DVD juga mahal, sayang sekali jika film yang Anda tonton tidak sesuai keinginan.

Sistem rujukan adalah program atau sistem penyaringan informasi yang merupakan solusi untuk masalah kelebihan informasi dengan  menyaring beberapa informasi penting dari  informasi yang tersedia dan bertindak berdasarkan minat, hubungan minat atau perilaku pengguna dalam suatu item. Sistem rekomendasi dirancang untuk memahami dan memprediksi preferensi pengguna berdasarkan perilaku pengguna [2]. Terdapat beberapa     metode yang dapat digunakan dalam membangun sebuah sistem rekomendasi antara lain *content based filtering*, *collaborative filtering*, dan *hybrid filtering*. *Collaborative filtering* menganalisis pengalaman dan peringkat pengguna sebelumnya dan menghubungkannya dengan pengguna lain [2].

Penulis ingin mempertimbangkan sistem rekomendasi film dengan mendeteksi kesamaan film yang ditonton berdasarkan rating yang diberikan oleh pengguna (*collaborative filtering*), kemudian dimungkinkan untuk mengurutkan berdasarkan rating film yang paling mirip dan akan menjadi rekomendasi film-film yang akan datang untuk ditonton. Jadi calon penonton tidak perlu lagi membuang waktu untuk mencari film satu per satu.

## **Business understanding**

Sebagai seorang Data Scientist, tentu Anda ingin memanfaatkan data tersebut untuk meningkatkan transaksi di perusahaan. 

### Problem Statements
Kembangkan sebuah sistem rekomendasi movie untuk menjawab permasalahan berikut:
*   Dengan data rating yang Anda miliki, bagaimana perusahaan dapat merekomendasikan movie lain yang mungkin disukai dan belum pernah ditonton oleh pengguna?  

### Goals
Untuk  menjawab pertanyaan tersebut, buatlah sistem rekomendasi dengan tujuan atau goals sebagai berikut:
*   Menghasilkan sejumlah rekomendasi movie yang sesuai dengan preferensi pengguna dan belum pernah ditonton sebelumnya dengan teknik *collaborative filtering*.

## **Data understanding**

Dataset yang digunakan berupa 2 file csv yang merupakan data rating movie dari pengguna dan data judul movie. Anda dapat mengunduh dataset ini melalui web Kaggle [berikut](https://www.kaggle.com/datasets/dev0914sharma/dataset).

### Data Loading

Supaya isi dataset lebih mudah dipahami, kita perlu melakukan proses loading data terlebih dahulu.

Data penilaian yang diberikan pengguna terdapat 100003 observasi dan data movie terdapat 1682 item.

![Screenshot (14)](https://user-images.githubusercontent.com/70586158/182511650-81316f63-af24-4b08-a2d3-0bf597bed1e3.png)

Perhatikanlah, data rating memiliki 100003 baris dan 4 kolom.

![Screenshot (15)](https://user-images.githubusercontent.com/70586158/182511783-4d80a7cb-8e66-4a99-a40e-40624ce3a41b.png)

Dari fungsi df.head(), kita dapat mengetahui bahwa data rating terdiri dari 5 kolom dengan lima kategori rating. Kolom-kolom tersebut antara lain:

* user_id, merupakan identitas pengguna.
* item_id, merupakan identitas movie.
* rating, merupakan data rating untuk movie.
* timestamp, merupakan data waktu saat pengguna memberikan rating pada movie.

![Screenshot (16)](https://user-images.githubusercontent.com/70586158/182511954-7574f3ca-b4a9-483b-8f69-5e345bdeed05.png)

Dari output di atas, diketahui bahwa nilai maksimum rating adalah 5 dan nilai minimumnya adalah 1. Artinya, skala rating berkisar antara 1 hingga 5. 

![Screenshot (17)](https://user-images.githubusercontent.com/70586158/182512995-0dfae968-f151-42fb-a926-6f101b1a03a3.png)

Untuk data movie memiliki 1682 judul movie dan 2 kolom yang terdiri dari id movie dan judul movie.

## **Data preparation**

Pada tahap ini, Anda perlu melakukan persiapan data untuk menyandikan (encode) fitur 'user_id' dan 'item_id' ke dalam indeks integer.

![Screenshot (19)](https://user-images.githubusercontent.com/70586158/182513318-8a88d32c-a4bc-4f05-b582-bca109446a60.png)

Selanjutnya, lakukan hal yang sama pada fitur 'item_id'.

![Screenshot (18)](https://user-images.githubusercontent.com/70586158/182513189-df665a02-db93-4941-87d2-5cb5ed028e3d.png)

### Membagi Data untuk Training dan Validasi

Pada tahap ini kita akan melakukan pembagian data menjadi data training dan validasi. Betul! Namun sebelumnya, acak datanya terlebih dahulu agar distribusinya menjadi random.

## **Modeling**

Pada tahap ini, kita akan mengembangkan model machine learning dengan menerapkan teknik collaborative filtering pada data.

### Proses Training

Pada tahap ini, model menghitung skor kecocokan antara pengguna dan movie dengan teknik embedding. Pertama, kita melakukan proses embedding terhadap data user dan movie. Selanjutnya, lakukan operasi perkalian dot product antara embedding user dan movie. Selain itu, kita juga dapat menambahkan bias untuk setiap user dan movie. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid.

Di sini, kita membuat class RecommenderNet dengan keras Model class. Kode class RecommenderNet ini terinspirasi dari tutorial dalam situs Keras dengan beberapa adaptasi sesuai kasus yang sedang kita selesaikan. Selanjutnya, lakukan proses compile terhadap model.

Model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. 

## **Evaluation**

Metrik evaluasi yang digunakan pada prediksi ini adalah root mean squared error (RMSE).

### Visualisasi Metrik

Untuk melihat visualisasi proses training, mari kita plot metrik evaluasi dengan matplotlib.

![download](https://user-images.githubusercontent.com/70586158/182511131-3ba68511-0bb4-4385-8dbb-eb19be5c5a61.png)

Perhatikanlah, proses training model konvergen pada epochs sekitar 80. Dari proses ini, kita memperoleh nilai error akhir sebesar sekitar 0.23 dan error pada data validasi sebesar 0.24. Nilai tersebut cukup bagus untuk sistem rekomendasi. 

### Mendapatkan Rekomendasi movie

![Screenshot (13)](https://user-images.githubusercontent.com/70586158/182511445-04e060ee-eea0-498c-9e30-f0e7c6ab5a64.png)

Sebagai contoh, hasil di atas adalah rekomendasi untuk user dengan id 286. Dari output tersebut, kita dapat membandingkan antara movie with high ratings from user dan Top 10 movie recommendation untuk user.

## **Kesimpulan**

Berdasarkan dari hasil  pengujian dan analisis dari implementasi yang telah dilakukan dapat disimpulkan bahwa sistem rekomendasi dengan teknik *collaborative filtering* sudah mencapai hasil yang baik dengan nilai error akhir sebesar sekitar 0.23 dan error pada data validasi sebesar 0.24.

## **Sumber Referensi**

[1] Tren Positif Film Indonesia, ‘Industri perfilman Indonesia semakin berkembang. Tren positip dan konsisten baik dari jumlah penonton maupun jumlah judul yang terdata sejak  tahun 2016-2018’, https://indonesia.go.id/ragam/seni/sosial/tren-positif-film-indonesia, 2019. (Diakses pada 2 Agustus 2022)

[2] Reddy, S. R. S. et al, ‘Content-Based Movie  Recommendation  System Using Genre Correlation’, in Satapathy, S. C., Bhateja,  V., and Das, S. (eds) Smart Intelligent Computing and Applications. Singapore: Springer Singapore, pp. 391–397, 2019.
