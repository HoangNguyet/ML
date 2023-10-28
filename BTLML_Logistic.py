from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
# Đọc dữ liệu từ tệp CSV
df = pd.read_csv('tieuduong_processed.csv')
X = np.array(df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values)
Y = np.array(df['Outcome'])

#chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state= 38)
# Thiết lập giao diện tkinter
form = Tk()
form.title("Dự đoán bệnh tiểu đường:")
form.geometry("750x500")

# Tiêu đề
label_title = Label(form, text="Dự đoán bệnh tiểu đường", font=("Comic Sans MS Bold", 19), fg="red")
label_title.grid(row=0, column=1, pady=10, sticky="e")

# Nhóm nhập thông tin để dự đoán
group_input = LabelFrame(form, text="Nhập thông tin để dự đoán")
group_input.grid(row=1, column=1, padx=50, pady=30)

label_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
input_entries = []

for row, feature in enumerate(label_features, start=1):
    label = Label(group_input, text=feature + ":")
    label.grid(row=row, column=1, pady=10, sticky="e")
    entry = Entry(group_input)
    entry.grid(row=row, column=2, padx=20)
    input_entries.append(entry)

# Nhóm hiển thị kết quả
group_output = LabelFrame(form, bd=0)
group_output.grid(row=1, column=2)

label_result = Label(group_output, text="Kết quả (Mắc(1)/Không mắc(0))", font=("Arial italic", 8))
label_result.grid(row=3, column=1, pady=10)

label_metrics = Label(group_output, text="Đánh giá dự đoán của mô hình:")
label_metrics.grid(row=0, column=1, padx=35, pady=20)

# Định nghĩa hàm dự đoán bằng SVM
def predict_svm():
    # Lấy giá trị từ các input
    input_data = [entry.get() for entry in input_entries]
    # Kiểm tra xem có giá trị trống không
    if any(value == '' for value in input_data):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        # Chuyển đổi giá trị input thành mảng numpy 2D
        input_array = np.array(input_data, dtype=float).reshape(1, -1)
        
        # Chuẩn hóa dữ liệu đầu vào
        input_scaled = scaler.transform(input_array)

        # Huấn luyện mô hình Hồi quy logistic
        model = LogisticRegression()
        model.fit(X_train,Y_train)
        # Dự đoán kết quả trên form
        prediction = model.predict(input_scaled)

        #Dự đoán kết quả dựa tên tập test
        y_pred = model.predict(X_test)
        # Tính toán các thông số
        accuracy = metrics.accuracy_score(Y_test,y_pred)
        precision = metrics.precision_score(Y_test,y_pred)
        recall = metrics.recall_score(Y_test,y_pred)
        f1_score = metrics.f1_score(Y_test,y_pred)
        
        # Hiển thị kết quả và đánh giá lên giao diện
        label_prediction.config(text=prediction[0])
        label_accuracy.config(text="Độ chính xác: {:.2%}".format(accuracy))
        label_precision.config(text="Precision: {:.2%}".format(precision))
        label_recall.config(text="Recall: {:.2%}".format(recall))
        label_f1_score.config(text="F1-Score: {:.2%}".format(f1_score))

# Nút dự đoán
button_predict = Button(group_output, text="Dự đoán", font=("Arial Bold", 9), fg="GreenYellow", bg="black", command=predict_svm)
button_predict.grid(row=2, column=1)

label_prediction = Label(group_output, text="...", font=("Arial Bold", 9), fg="white", bg="SlateGray4")
label_prediction.grid(row=4, column=1)

label_accuracy = Label(group_output, text="", font=("Arial Bold", 9), fg="black")
label_accuracy.grid(row=5, column=1)

label_precision = Label(group_output, text="", font=("Arial Bold", 9), fg="black")
label_precision.grid(row=6, column=1)

label_recall = Label(group_output, text="", font=("Arial Bold", 9), fg="black")
label_recall.grid(row=7, column=1)

label_f1_score = Label(group_output, text="", font=("Arial Bold", 9), fg="black")
label_f1_score.grid(row=8, column=1)
form.mainloop()
