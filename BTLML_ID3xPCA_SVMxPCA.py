from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


#tính tỉ lệ dự đoán đúng
def tyledung(y_test,y_pred):
    d = 0
    for i in range(len(y_pred)):
        if(y_pred[i]==y_test[i]):
            d = d+1
    return (d/len(y_pred))

df = pd.read_csv('tieuduong_processed.csv')
X =  np.array(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].values)
Y = np.array(df['Outcome'])
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#Fit dữ liệu vào
scaler.fit(X)
#Thực hiện transform scale
scaler_x = scaler.transform(X)

max_id3 = 0     #lưu giá trị tỉ lệ đúng lớn nhất trong các mô hình khi sd ID3
max_svc = 0
for j in range(1,9):
    pca = PCA(n_components=j)
    pca.fit(X)

    X_bar = pca.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_bar, Y, test_size=0.3 , shuffle = False)

    #id3
    id3 = DecisionTreeClassifier(criterion='entropy', max_depth=None)
    id3.fit(X_train, y_train)
    y_pred_id3 = id3.predict(X_test)
    rate_id3 = tyledung(y_test,y_pred_id3)

    #svm
    svc = SVC(kernel = 'linear')
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    rate_svc = tyledung(y_test,y_pred_svc)

    if(rate_id3 >max_id3):
        num_pca_id3 = j     #lưu lại số thuộc tính tốt nhất
        pca_best_id3 = pca  #lưu lại mô hình pca tốt nhất
        max_id3 = rate_id3      #lưu lại tỉ lệ dự đoán đúng của mô hình tốt nhất
        modelmax_id3 = id3  #mô hình có tỉ lệ đúng lớn nhất
        y_pred_best_id3 = y_pred_id3

    if(rate_svc >max_svc):
        num_pca_svc = j     #lưu lại số thuộc tính tốt nhất
        pca_best_svc = pca  #lưu lại mô hình pca tốt nhất
        max_svc = rate_svc      #lưu lại tỉ lệ dự đoán đúng của mô hình tốt nhất
        modelmax_svc = svc  #mô hình có tỉ lệ đúng lớn nhất
        y_pred_best_svc = y_pred_svc
    print ("Số thuộc tính tốt nhất: ", j)
    print("- Tỉ lệ dự đoán đúng của thuật toán ID3: ",rate_id3 )
    print("- Tỉ lệ dự đoán đúng của thuật toán SVM: ",rate_svc )

#form
form = Tk()             #tạo ra cửa sổ gán vào biến form
form.title("Dự đoán bệnh tiểu đường:") #thay đổi tiêu đề cửa sổ
form.geometry("760x650")   #kích thước cửa sổ

lable_dudoan = Label(form, text = "Dự đoán bệnh tiểu đường", font=("Arial", 20), fg = "brown").grid(row = 0, column = 1, pady = 10, sticky="e")

lable_ten = Label(form, text = "Nhập thông tin để dự đoán:", font=("Arial Bold", 10), fg="red")
lable_ten.grid(row = 1, column = 1, pady = 10, sticky='e')

lable_pre = Label(form, text = " Pregnancies:   ")
lable_pre.grid(row = 2, column = 1, pady = 10, sticky='e')
textbox_pre = Entry(form)
textbox_pre.grid(row = 2, column = 2)

lable_glu = Label(form, text = "Glucose:   ")
lable_glu.grid(row = 3, column = 1, pady = 10, sticky='e')
textbox_glu = Entry(form)
textbox_glu.grid(row = 3, column = 2)

lable_blood = Label(form, text = "BloodPressure:   ")
lable_blood.grid(row = 4, column = 1,pady = 10, sticky='e')
textbox_blood = Entry(form)
textbox_blood.grid(row = 4, column = 2)

lable_skin = Label(form, text = "SkinThickness:   ")
lable_skin.grid(row = 5, column = 1, pady = 10, sticky='e')
textbox_skin = Entry(form)
textbox_skin.grid(row = 5, column = 2)

lable_insu = Label(form, text = "Insulin:   ")
lable_insu.grid(row = 6, column = 1, pady = 10 , sticky='e')
textbox_insu = Entry(form)
textbox_insu.grid(row = 6, column = 2)

lable_bmi = Label(form, text = "BMI:   ")
lable_bmi.grid(row = 7, column = 1, pady = 10,sticky='e' )
textbox_bmi = Entry(form)
textbox_bmi.grid(row = 7, column = 2)

lable_Dia = Label(form, text = "DiabetesPedigreeFunction:   ")
lable_Dia.grid(row = 8, column = 1, pady = 10,sticky='e' )
textbox_Dia = Entry(form)
textbox_Dia.grid(row = 8, column = 2)

lable_age = Label(form, text = "Age:   ")
lable_age.grid(row = 9, column = 1, pady = 10, sticky="e" )
textbox_age = Entry(form)
textbox_age.grid(row = 9, column = 2)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ID3

#dudoanid3
lb_id3 = Label(form)
lb_id3.grid(column=1, row=10)
lb_id3.configure(text="\n\nTỉ lệ dự đoán đúng của ID3: "+'\n'
                           +"Precision: "+str(precision_score(y_test, y_pred_best_id3, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test, y_pred_best_id3, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test, y_pred_best_id3, average='macro')*100)+"%"+'\n'
                           +"Accuracy: "+str(max_id3*100)+"%")
                        

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
        X_dudoan_bar = pca_best_id3.transform(X_dudoan)
        y_kqua = modelmax_id3.predict(X_dudoan_bar)
        lb_pred_id3.configure(text= y_kqua)

button_1 = Button(form, text = 'Kết quả dự đoán theo ID3',font=("Arial Bold", 9), fg = "black", bg = "green", command = dudoanID3)
button_1.grid(row = 11, column = 1, pady = 20)
lb_pred_id3 = Label(form, text="...")
lb_pred_id3.grid(column=1, row=12)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SVM

#dudoansvm
lb_svc = Label(form)
lb_svc.grid(column=3, row=10)
lb_svc.configure(text="\n\nTỉ lệ dự đoán đúng của SVM: "+'\n'
                           +"Precision: "+str(precision_score(y_test,  y_pred_best_svc, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test,  y_pred_best_svc, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test,  y_pred_best_svc, average='macro')*100)+"%"+'\n'
                           +"Accuracy: "+str(max_svc*100)+"%")

#hàm dự đoán giá trị theo SVM
def dudoanSVM():
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
        X_dudoan_bar = pca_best_svc.transform(X_dudoan)
        y_kqua = modelmax_svc.predict(X_dudoan_bar)
        lb_pred_svm.configure(text= y_kqua)

button_3 = Button(form, text = 'Kết quả dự đoán theo SVM',font=("Arial Bold", 9), fg = "black", bg = "green", command = dudoanSVM)
button_3.grid(row = 11, column = 3, pady = 20)
lb_pred_svm = Label(form, text="...")
lb_pred_svm.grid(column=3, row=12)


form.mainloop() #hiển thị cửa sổ

