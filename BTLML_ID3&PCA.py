from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

df = pd.read_csv('tieuduong_processed.csv')
X =  np.array(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].values)
Y = np.array(df['Outcome'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#Fit dữ liệu vào
scaler.fit(X)
#Thực hiện transform scale
scaler_x = scaler.transform(X)

#Hàm tính tỉ lệ dự đoán đúng
def RateRating(Y_pred):
	countPredictTrue = 0
	for i in range(len(Y_pred)):
		if(Y_pred[i] == Y_test[i]):
			countPredictTrue = countPredictTrue + 1
		rate = countPredictTrue / len(Y_pred)
	return rate

#Duyệt tìm mô hình tốt nhất
max_id3 = 0
for i in range(1,9): #duyệt qua từ 1-8 (i là số components PCA), 
    pca = PCA(n_components = i) #tạo mô hình pca với số thành phần chính được thiết lập là i
    pca.fit(scaler_x) #Huấn luyện mô hình với tập dữ liệu X
    X_bar = pca.transform(scaler_x) # Giảm chiều dữ liệu
    X_train, X_test, Y_train, Y_test = train_test_split(X_bar, Y, test_size=0.3,shuffle=False)
    #với mỗi i ta xây dựng 1 cây quyết định
    #id3
    id3 = DecisionTreeClassifier(criterion='entropy')
    id3.fit(X_train, Y_train) # huấn luyện cây 
    y_pred_id3 = id3.predict(X_test) #Tính kết quả y_test dựa vào x_test dựa vào mô hình vừa huấn luyện
    if(RateRating(y_pred_id3) > max_id3):
        num_pca_id3 = i #Lưu thuộc tính tốt nhất
        pca_best_id3 = pca #lưu mô hình pca tốt nhất
        max_id3 = RateRating(y_pred_id3)
        modelmax_id3 = id3 
        y_pred_best_id3 = y_pred_id3
#form
form = Tk()          
form.title("Dự đoán bệnh tiểu đường:") 
form.geometry("750x500")

lable_dudoan = Label(form, text = "Dự đoán bệnh tiểu đường", font=("Arial", 20), fg = "brown").grid(row = 0, column = 1, pady = 10, sticky="e")

group1 = LabelFrame(form, text="Nhập thông tin để dự đoán")
group1.grid(row=1, column=1, padx=50, pady=30)
group2 = LabelFrame(form, bd=0)
group2.grid(row=1, column=2)
group3 = LabelFrame(group2, text="Đánh giá mô hình được chọn:")
group3.grid(row=1, column=1, pady=20)

lable_pre = Label(group1, text = " Pregnancies:").grid(row = 1, column = 1, pady = 10,sticky="e")
textbox_pre = Entry(group1)
textbox_pre.grid(row = 1, column = 2, padx = 20)

lable_glu = Label(group1, text = "Glucose:").grid(row = 2, column = 1, pady = 10,sticky="e")
textbox_glu = Entry(group1)
textbox_glu.grid(row = 2, column = 2)

lable_blood = Label(group1, text = "BloodPressure:").grid(row = 3, column = 1,pady = 10,sticky="e")
textbox_blood = Entry(group1)
textbox_blood.grid(row = 3, column = 2)

lable_skin = Label(group1, text = "SkinThickness:").grid(row = 4, column = 1,pady = 10,sticky="e")
textbox_skin = Entry(group1)
textbox_skin.grid(row = 4, column = 2)

lable_insu = Label(group1, text = "Insulin:").grid(row = 5, column = 1, pady = 10,sticky="e")
textbox_insu = Entry(group1)
textbox_insu.grid(row = 5, column = 2)

lable_bmi = Label(group1, text = "BMI:").grid(row = 6, column = 1, pady = 10,sticky="e")
textbox_bmi = Entry(group1)
textbox_bmi.grid(row = 6, column = 2)

lable_Dia = Label(group1, text = "DiabetesPedigreeFunction:").grid(row = 7, column = 1, pady = 10,sticky="e")
textbox_Dia = Entry(group1)
textbox_Dia.grid(row = 7, column = 2)

lable_age = Label(group1, text = "Age:").grid(row = 8, column = 1, pady = 10,sticky="e")
textbox_age = Entry(group1)
textbox_age.grid(row = 8, column = 2)

lable_ketqua = Label(group2, text = "Kết quả(true/false)", font=("Arial italic", 8)).grid(row = 3, column = 1, pady = 10)
#lable_ketqua.grid(row=3, column=1, pady=10)
#Đánh giá độ đo
lb_id3 = Label(group3)
lb_id3.grid(row=0, column=1, padx = 35, pady = 20)
lb_id3.configure(text=  "Precision: "+str(metrics.precision_score(Y_test, y_pred_best_id3, average='macro')*100)+"%"+'\n'
                        +"\nRecall: "+str(metrics.recall_score(Y_test, y_pred_best_id3, average='macro')*100)+"%"+'\n'
                        +"\nF1-score: "+str(metrics.f1_score(Y_test, y_pred_best_id3, average='macro')*100)+"%"+'\n'
                        +"\nAccuracy: "+str(metrics.accuracy_score(Y_test, y_pred_best_id3)*100)+"%")
lb_num = Label(group3)
lb_num.grid(row=2, column=1, padx = 35, pady = 20)
lb_num.configure(text = "Số chiều tốt nhất: "+str( num_pca_id3))
#hàm dự đoán giá trị theo ID3
def dudoanID3():
    Pregnancies = textbox_pre.get()
    Glucose = textbox_glu.get()
    BloodPressure = textbox_blood.get()
    SkinThickness = textbox_skin.get()
    Insulin = textbox_insu.get()
    BMI = textbox_bmi.get()
    DiabetesPedigreeFunction = textbox_Dia.get()
    Age = textbox_age.get()
    if((Pregnancies == '') or (Glucose == '') or (BloodPressure == '') or (SkinThickness == '') or (Insulin == '') or (BMI == '') or (DiabetesPedigreeFunction == '') or (Age == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction, Age]).reshape(1, -1)
        X_dudoan_bar = pca_best_id3.transform(X_dudoan) # Dùng mô hình pca tốt nhất để làm giảm chiều dữ liệu
        y_kqua = modelmax_id3.predict(X_dudoan_bar) # Dùng mô hình id3 tốt nhất để dự đoán kết quả đầu vào
        lb_pred_id3.configure(text= y_kqua[0])

button_1 = Button(group2, text = 'Kết quả dự đoán', font=("Arial Bold", 9), fg = "black", bg = "green", command = dudoanID3)
button_1.grid(row = 2, column = 1)
lb_pred_id3 = Label(group2, text="...", font=("Arial Bold", 9), fg = "white", bg = "SlateGray4")
lb_pred_id3.grid(row=4, column=1)

form.mainloop() #chạy giao diện chính chờ người dùng tương tác


