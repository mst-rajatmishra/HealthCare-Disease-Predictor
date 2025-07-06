from tkinter import *
import numpy as np
import pandas as pd
# from gui_stuff import *

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

# print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
# ------------------------------------------------------------------------------------------------------

def DecisionTree():

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")


def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")


def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")

# gui_stuff------------------------------------------------------------------------------------

root = Tk()
root.title("HealthCare Disease Predictor")
root.configure(background='#F5F7FA')  # Brighter background
root.geometry('900x650')
root.resizable(False, False)

# entry variables
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
Name = StringVar()

# Heading
w2 = Label(root, justify=LEFT, text="HealthCare Disease Predictor", fg="#1565C0", bg="#F5F7FA")
w2.config(font=("Segoe UI", 28, "bold"))
w2.grid(row=1, column=0, columnspan=4, padx=30, pady=30)

# labels
label_font = ("Segoe UI", 13, "bold")
label_bg = "#E3F2FD"
label_fg = "#1565C0"

NameLb = Label(root, text="Name of the Patient", fg=label_fg, bg=label_bg, font=label_font)
NameLb.grid(row=6, column=0, pady=15, sticky=W, padx=30)

S1Lb = Label(root, text="Symptom 1", fg=label_fg, bg=label_bg, font=label_font)
S1Lb.grid(row=7, column=0, pady=10, sticky=W, padx=30)

S2Lb = Label(root, text="Symptom 2", fg=label_fg, bg=label_bg, font=label_font)
S2Lb.grid(row=8, column=0, pady=10, sticky=W, padx=30)

S3Lb = Label(root, text="Symptom 3", fg=label_fg, bg=label_bg, font=label_font)
S3Lb.grid(row=9, column=0, pady=10, sticky=W, padx=30)

S4Lb = Label(root, text="Symptom 4", fg=label_fg, bg=label_bg, font=label_font)
S4Lb.grid(row=10, column=0, pady=10, sticky=W, padx=30)

S5Lb = Label(root, text="Symptom 5", fg=label_fg, bg=label_bg, font=label_font)
S5Lb.grid(row=11, column=0, pady=10, sticky=W, padx=30)

algo_label_font = ("Segoe UI", 12, "bold")
algo_label_bg = "#B2EBF2"
algo_label_fg = "#006064"

lrLb = Label(root, text="DecisionTree", fg=algo_label_fg, bg=algo_label_bg, font=algo_label_font)
lrLb.grid(row=15, column=0, pady=10,sticky=W, padx=30)

destreeLb = Label(root, text="RandomForest", fg=algo_label_fg, bg=algo_label_bg, font=algo_label_font)
destreeLb.grid(row=17, column=0, pady=10, sticky=W, padx=30)

ranfLb = Label(root, text="NaiveBayes", fg=algo_label_fg, bg=algo_label_bg, font=algo_label_font)
ranfLb.grid(row=19, column=0, pady=10, sticky=W, padx=30)

# entries
OPTIONS = sorted(l1)

entry_font = ("Segoe UI", 12)
entry_bg = "#FFFFFF"
entry_fg = "#222831"

NameEn = Entry(root, textvariable=Name, font=entry_font, bg=entry_bg, fg=entry_fg, relief=GROOVE, bd=2, highlightbackground="#90CAF9", highlightcolor="#1976D2", highlightthickness=2)
NameEn.grid(row=6, column=1, padx=20, pady=5, sticky=W)

S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.config(font=entry_font, bg=entry_bg, fg=entry_fg, relief=GROOVE, bd=2, width=22, highlightbackground="#90CAF9")
S1En.grid(row=7, column=1, padx=20, pady=5, sticky=W)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.config(font=entry_font, bg=entry_bg, fg=entry_fg, relief=GROOVE, bd=2, width=22, highlightbackground="#90CAF9")
S2En.grid(row=8, column=1, padx=20, pady=5, sticky=W)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.config(font=entry_font, bg=entry_bg, fg=entry_fg, relief=GROOVE, bd=2, width=22, highlightbackground="#90CAF9")
S3En.grid(row=9, column=1, padx=20, pady=5, sticky=W)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.config(font=entry_font, bg=entry_bg, fg=entry_fg, relief=GROOVE, bd=2, width=22, highlightbackground="#90CAF9")
S4En.grid(row=10, column=1, padx=20, pady=5, sticky=W)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.config(font=entry_font, bg=entry_bg, fg=entry_fg, relief=GROOVE, bd=2, width=22, highlightbackground="#90CAF9")
S5En.grid(row=11, column=1, padx=20, pady=5, sticky=W)

button_font = ("Segoe UI", 12, "bold")
button_bg = "#1976D2"
button_fg = "#FFFFFF"

style = {"font": button_font, "bg": button_bg, "fg": button_fg, "activebackground": "#64B5F6", "activeforeground": "#222831", "bd": 3, "relief": RAISED, "width": 16, "height": 1}

dst = Button(root, text="DecisionTree", command=DecisionTree, **style)
dst.grid(row=8, column=3, padx=30, pady=5)

rnf = Button(root, text="Randomforest", command=randomforest, **style)
rnf.grid(row=9, column=3, padx=30, pady=5)

lr = Button(root, text="NaiveBayes", command=NaiveBayes, **style)
lr.grid(row=10, column=3, padx=30, pady=5)

# textfields
text_font = ("Segoe UI", 12, "bold")
text_bg = "#E3F2FD"
text_fg = "#1565C0"

t1 = Text(root, height=1, width=45, bg=text_bg, fg=text_fg, font=text_font, relief=GROOVE, bd=2)
t1.grid(row=15, column=1, padx=20, pady=5, sticky=W)

t2 = Text(root, height=1, width=45, bg=text_bg, fg=text_fg, font=text_font, relief=GROOVE, bd=2)
t2.grid(row=17, column=1, padx=20, pady=5, sticky=W)

t3 = Text(root, height=1, width=45, bg=text_bg, fg=text_fg, font=text_font, relief=GROOVE, bd=2)
t3.grid(row=19, column=1, padx=20, pady=5, sticky=W)

root.mainloop()
