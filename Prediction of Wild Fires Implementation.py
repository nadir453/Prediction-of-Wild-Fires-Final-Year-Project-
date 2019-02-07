import tkinter
from tkinter import *
from tkinter import messagebox   
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import cross_validation 
from sklearn import tree
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
 # Visualising the Training set results
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn import neighbors, datasets, preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score, v_measure_score
from sklearn import neighbors, datasets, preprocessing
from sklearn.preprocessing import LabelEncoder,Normalizer,Binarizer, Imputer,PolynomialFeatures,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from tkinter.font import Font
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import pylab as pl
from sklearn import neighbors
from math import sqrt
from sklearn import svm
from sklearn.preprocessing import Normalizer
from sklearn import tree



def Load_Dataset():

   
    
    print("New File!")
    dataset = pd.read_csv("D:\Copy of Modified Wildfire Dataset for Analysis.csv",encoding = "ISO-8859-1")

    lb_Main_headng =   tkinter.Label(root,text="Dataset OverView and Feature Importance ",font=("Georgia", "26", "bold"), fg = "black")
    lb_Main_headng.place(x=400,y=80)
    

    f = Font(lb_Main_headng, lb_Main_headng.cget("font")) #To make the font underline in the Heading we have to use font class that has underline attribute
    f.configure(underline= 1)

    lb_cnt_datset = tkinter.Label(root,text="Total Records:")
    lb_cnt_datset.place(x=70,y =210)
    
    lb_tst_datset = tkinter.Label(root,text="Test Set:")
    lb_tst_datset.place(x=70,y =250)
    
    
    
    Btn_loadfile = tkinter.Button(root,text="Load Dataset",fg="black",command=Load_file)
    Btn_loadfile.place(x=200,y=350,width=150,height=80)

    #Btn_Preview = tkinter.Button(root,text="Preview ",fg="black",command=Preview_file)
    #Btn_Preview.place(x=240,y=350,width=100,height=60)

    Txt_Dataset_cnt  =   Text(root)
    Txt_Dataset_cnt.place(x=200,y=210,width=150,height=30)
    
    
    Txt_tst_cnt  =   Text(root)
    Txt_tst_cnt.place(x=200,y=250,width=150,height=30)
   
    
    print("The total rows in Dataset is:",len(dataset))
    col_lstbox   =   Listbox(root)
    col_lstbox.place(x=400,y=210,width=140,height=240)
    

    col_dty_lstbox   =   Listbox(root)
    col_dty_lstbox.place(x=500,y=210,width=100,height=240)
    
    lb_col_name = tkinter.Label(root,text="Columns names:")
    lb_col_name.place(x=400,y =190)

    lb_Dtype_name = tkinter.Label(root,text="Data Type:")
    lb_Dtype_name.place(x=540,y =190)

  
   
    
   
    
def Load_file():
    dataset = pd.read_csv("D:\Copy of Modified Wildfire Dataset for Analysis.csv",encoding = "ISO-8859-1")
    messagebox.showinfo("Title","Wild Fire Dataset Loaded Successfully"  )

    Txt_Dataset_cnt  =   Text(root)
    Txt_Dataset_cnt.place(x=200,y=210,width=150,height=30)
    Txt_Dataset_cnt.insert(INSERT,"{:,}".format(len(dataset)))
    Txt_Dataset_cnt.tag_add("start", "1.0", END)
    Txt_Dataset_cnt.tag_config("start", font=("Georgia", "12", "bold"))
    
    Txt_tst_cnt  =   Text(root)
    Txt_tst_cnt.place(x=200,y=250,width=150,height=30)
    Txt_tst_cnt.insert(INSERT,"{:,}".format(round(len(dataset)* 0.3),0))
    Txt_tst_cnt.tag_add("start", "1.0", END)
    Txt_tst_cnt.tag_config("start", font=("Georgia", "12", "bold"))

    col_lstbox   =   Listbox(root)
    col_lstbox.place(x=400,y=210,width=140,height=240)
    colu = dataset.columns

    col_dty_lstbox   =   Listbox(root)
    col_dty_lstbox.place(x=500,y=210,width=100,height=240)
    col_dtype = dataset.dtypes
    
    dataset = dataset[['FIRE_YEAR', 'DISCOVERY_DOY' ,'DISCOVERY_TIME','CONT_DOY','CONT_TIME','FIRE_SIZE','LATITUDE','LONGITUDE','OWNER_CODE','REGION']]
    
    for x in colu:
       col_lstbox.insert(0,x)

    for d in col_dtype:
        print("Columns datatypes are  :",d)
        col_dty_lstbox.insert(0,d)

    
    Target_lstbox   =   Listbox(root)
    Target_lstbox.place(x=700,y=210,width=150,height=240)

    lb_uniq_reg = tkinter.Label(root,text="US Region:")
    lb_uniq_reg.place(x=700,y =190)

    lb_inst = tkinter.Label(root,text="Instance:")
    lb_inst.place(x=810,y =190)

    lb_feature = tkinter.Label(root,text="Feature:")
    lb_feature.place(x=950,y =190)

    lb_Feat_score =   tkinter.Label(root  ,text="Score")
    lb_Feat_score.place(x=1050,y=190)
    
    
    Region_Unique = pd.unique(dataset['REGION'])
    print("The Unique Regions are ", Region_Unique)

    for tr in Region_Unique:
         Target_lstbox.insert(0,tr)
         
    Target_inst_lstbox   =   Listbox(root)
    Target_inst_lstbox.place(x=800,y=210,width=80,height=240)

    region_inst = dataset.groupby('REGION').size()
    for rs in region_inst:
        Target_inst_lstbox.insert(0,format(rs,',d'))

    fea_lstbox   =   Listbox(root  )
    fea_lstbox.place(x=950,y=210,width=150,height=240)

    feat_score = Listbox(root )
    feat_score.place(x=1050,y=210,width=80,height=240)

    X= dataset.drop('REGION',axis=1)
    y= dataset['REGION']
    
    enc = LabelEncoder()

    y = enc.fit_transform(y)
    colu = dataset.columns
    

    model_import = ExtraTreesClassifier()
    model_import.fit(X,y)

    print("Features sorted by their score:")
    feat_import= reversed(sorted(zip(map(lambda x: round(x, 4), model_import.feature_importances_),  colu), reverse=True))
    #print(sorted(zip(map(lambda x: round(x, 4), model_import.feature_importances_),  colu), reverse=True))
    print("feature importance",feat_import)

    for x in feat_import:
        cnt = 0 
        for xr in x:
            print("VAlue of xr",cnt)
            if cnt == 0:
               feat_score.insert(0,xr)
            if cnt ==1:
               fea_lstbox.insert(0,xr)
            cnt+=1
    
    
  

def Decision_Tree():

    
    Decs = Tk()
    Decs.title("Decision Tree")
    dataset = pd.read_csv("D:\Copy of Modified Wildfire Dataset for Analysis.csv",encoding = "ISO-8859-1")
    lb_Decs_headng =   tkinter.Label(Decs,text="Decision Tree",font=("Georgia", "26", "bold"), fg = "black")
    lb_Decs_headng.place(x=450,y=60)
    X= 0

    #Btn_Genrate_Decson = tkinter.Button(Decs,text="Generate Output",fg="black",command=Generate_Decsion_Output)
    #Btn_Genrate_Decson.place(x=155,y=400,width=140,height=60)

    lb_accur_score =   tkinter.Label(Decs,text="Accuracy Score:")
    lb_accur_score.place(x=150,y=150)

    lb_abs_error =   tkinter.Label(Decs,text="Mean Absolute error:")
    lb_abs_error.place(x=300,y=150)

    lb_mean_score =   tkinter.Label(Decs,text="RMSE:")
    lb_mean_score.place(x=450,y=150)

#    lb_rsqre =   tkinter.Label(Decs,text="Mean R-Square Error:")
#    lb_rsqre.place(x=600,y=150)

    lb_f1_score =   tkinter.Label(Decs,text="F1 Score:")
    lb_f1_score.place(x=750,y=150)

    lb_recall_score =   tkinter.Label(Decs,text="Recall Score:")
    lb_recall_score.place(x=860,y=150)

    lb_precson_score =   tkinter.Label(Decs,text="Precision Score:")
    lb_precson_score.place(x=1000,y=150)

    Txt_accur  =   Text(Decs)
    Txt_accur.place(x=150,y=180,width=150,height=30)

    Txt_abs_error  =   Text(Decs)
    Txt_abs_error.place(x=300,y=180,width=150,height=30)

    Txt_mean_sqre  =   Text(Decs)
    Txt_mean_sqre.place(x=450,y=180,width=150,height=30)

#    Txt_rsqre  =   Text(Decs)
#    Txt_rsqre.place(x=600,y=180,width=150,height=30)

    Txt_f1_score  =   Text(Decs)
    Txt_f1_score.place(x=740,y=180,width=150,height=30)

    Txt_recall_score  =   Text(Decs)
    Txt_recall_score.place(x=850,y=180,width=150,height=30)

    Txt_precson_score  =   Text(Decs)
    Txt_precson_score.place(x=980,y=180,width=150,height=30)
    
    lb_accur_score =   tkinter.Label(Decs,text="Confusion Matrix:",font=("Georgia", "18", "bold"), fg = "black")
    lb_accur_score.place(x=150,y= 250)


    Region_Unique = pd.unique(dataset['REGION'])
    print("The Unique Regions are ",Region_Unique)

#    dataset = dataset[['FIRE_YEAR', 'DISCOVERY_DOY' ,'DISCOVERY_TIME','CONT_DOY','CONT_TIME','FIRE_SIZE','LATITUDE','LONGITUDE','OWNER_CODE','REGION_CODE']]
#    #print(dataset)
#
#     
#    dataset = dataset.dropna()
#    print(1)
#
#    
#    y = dataset['REGION_CODE']
#    
#    print("Target dataset",y)
#
#    X= dataset.drop('REGION_CODE',axis=1)
    print("Non predictor variables",X)
    X = dataset.iloc[:, [0,1,2,3,5,6,7,9,10,11]].values
    y = dataset.iloc[:, 14].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=100)
  
    
   #Feature Scaling
  
#    sc_X = StandardScaler()
#    X_train = sc_X.fit_transform(X_train)
#    X_test = sc_X.transform(X_test)


    model = DecisionTreeClassifier(criterion = 'entropy', random_state =0 )


    #model = tree.DecisionTreeClassifier(criterion = "entropy", random_state = 100)
   

    print("The model for decsision tree generated is ",model)

    fit = model.fit(X_train, y_train)
    print("The model fit for our training data",fit)


    y_predict = model.predict(X_test)



    acurcy_score =accuracy_score(y_test, y_predict)

    print("\n The accuracy score ",acurcy_score)



    print("The data for y_test is array for Region is ",y_test )

    print("The data for y_predict is array for Region is ",y_predict )

    print("Mean Absolute error is ",mean_absolute_error(y_test,y_predict ))
    print("Mean Squared Error is ",mean_squared_error(y_test,y_predict))
    print("The Recall or Sensitivity is ",recall_score(y_test, y_predict,average='weighted'))
    print("The Precision Score is ", precision_score(y_test, y_predict,average='weighted')     )
    

    

    Txt_abs_error.insert(INSERT,"%.3f" % mean_absolute_error(y_test,y_predict )          )
    Txt_abs_error.tag_add("start", "1.0", END)
    Txt_abs_error.tag_config("start", font=("Georgia", "12", "bold"))

    Txt_mean_sqre.insert(INSERT,"%.3f" % mean_squared_error(y_test,y_predict)          )
    Txt_mean_sqre.tag_add("start", "1.0", END)
    Txt_mean_sqre.tag_config("start", font=("Georgia", "12", "bold"))


#    Txt_rsqre.insert(INSERT,"%.1f" % round(r2_score(y_test,y_predict )          ))
#    Txt_rsqre.tag_add("start", "1.0", END)
#    Txt_rsqre.tag_config("start", font=("Georgia", "12", "bold"))



    confusionMatrix = confusion_matrix(y_test, y_predict)
    print("The Confusion Matrix is",confusion_matrix(y_test, y_predict))
    print("The confusionMatrix",confusionMatrix)
    
    # True Positive values
    lb_NE_TP =   tkinter.Label(Decs,text="TP" ,font=("Georgia", "14", "bold"), fg = "black")
    lb_NE_TP.place(x=200,y=310)

    Txt_NE_TP  =   Text(Decs)
    Txt_NE_TP.place(x=150,y=340,width=100,height=50)
    
    lb_West_Regon =   tkinter.Label(Decs,text="Western" ,font=("Georgia", "10", "bold"), fg = "black")
    lb_West_Regon.place(x=60,y=360)
    
    lb_West_Regon =   tkinter.Label(Decs,text="Mid Western" ,font=("Georgia", "10", "bold"), fg = "black")
    lb_West_Regon.place(x=40,y=400)
    
    
    
    Txt_NE_TP .insert(INSERT,       confusionMatrix[0, 0] )
    Txt_NE_TP .tag_add("start", "1.0", END)
    Txt_NE_TP.tag_config("start", font=("Georgia", "12", "bold"))
    
    #True Negative
    
    lb_NE_TN =   tkinter.Label(Decs,text="TN" ,font=("Georgia", "14", "bold"), fg = "black")
    lb_NE_TN.place(x=280,y=310)

    Txt_NE_TN  =   Text(Decs)
    Txt_NE_TN.place(x=250,y=340,width=100,height=50)
    
    Txt_NE_TN .insert(INSERT,       confusionMatrix[0, 1] )
    Txt_NE_TN .tag_add("start", "1.0", END)
    Txt_NE_TN.tag_config("start", font=("Georgia", "12", "bold"))
    
    #False Positive
    Txt_NE_FP  =   Text(Decs)
    Txt_NE_FP.place(x=150,y=390,width=100,height=50)
    
    lb_NE_FP =   tkinter.Label(Decs,text="FP" ,font=("Georgia", "14", "bold"), fg = "black")
    lb_NE_FP.place(x=180,y=440)
    
    Txt_NE_FP .insert(INSERT,       confusionMatrix[1, 0] )
    Txt_NE_FP .tag_add("start", "1.0", END)
    Txt_NE_FP.tag_config("start", font=("Georgia", "12", "bold"))
    
    # False Negative
    Txt_NE_FN  =   Text(Decs)
    Txt_NE_FN.place(x=250,y=390,width=100,height=50)
    
    lb_NE_FN =   tkinter.Label(Decs,text="FN" ,font=("Georgia", "14", "bold"), fg = "black")
    lb_NE_FN.place(x=280,y=440)
    
    Txt_NE_FN.insert(INSERT,       confusionMatrix[1, 1] )
    Txt_NE_FN.tag_add("start", "1.0", END)
    Txt_NE_FN.tag_config("start", font=("Georgia", "12", "bold"))
    
    
    
    TP= confusionMatrix[0, 0] 
    FN = confusionMatrix[1, 1]
    FP = confusionMatrix[1, 0]
    TN = confusionMatrix[0, 1]
    
    print("TP",TP)
    print("FN",FN)
    accur_score = (TP + FN) / 16904
   
    Txt_accur.insert(INSERT,"%.2f" %  accur_score)
    Txt_accur.tag_add("start", "1.0", END)
    Txt_accur.tag_config("start", font=("Georgia", "12", "bold"))
    
    print("\n The accuracy score ",accur_score)
    
    precson = TP /(TP + FP)
    print("Precision",precson )
    
    Txt_precson_score.insert(INSERT,"%.3f" % precson      )
    Txt_precson_score.tag_add("start", "1.0", END)
    Txt_precson_score.tag_config("start", font=("Georgia", "12", "bold"))
    
    recall = TP / (TP + FN)
    
    Txt_recall_score.insert(INSERT,"%.3f" % recall )
    Txt_recall_score.tag_add("start", "1.0", END)
    Txt_recall_score.tag_config("start", font=("Georgia", "12", "bold"))
    
    f1_score = 2 / ((1 / precson) + (1 /  recall))
    
    Txt_f1_score.insert(INSERT,"%.3f" % f1_score)
    Txt_f1_score.tag_add("start", "1.0", END)
    Txt_f1_score.tag_config("start", font=("Georgia", "12", "bold"))

    
   


#    print("The V measure score for Decision tree  is ",v_measure_score(y_test,y_predict))
#    
#    print(classification_report(y_test, y_predict))
#    print("\n\n")
#    print("F1 score",f1_score(y_test, y_predict,average='weighted'))
#    print("Recall score",recall_score(y_test, y_predict,average='weighted'))
#    print("Precision score",precision_score(y_test, y_predict,average='weighted'))

   
   # Visualising the Training set results

  

   
   
   

def Classification():

    Cls = Tk()
    Cls.title("Classification")
    dataset = pd.read_csv("D:\Copy of Modified Wildfire Dataset for Analysis.csv",encoding = "ISO-8859-1")
    lb_Decs_headng =   tkinter.Label(Cls,text="Classification ",font=("Georgia", "26", "bold"), fg = "black")
    lb_Decs_headng.place(x=450,y=60)
    X= 0


    #Btn_Generte_Diagram = tkinter.Button(Cls,text="Generate Diagram",fg="black",command= Generte_Diagram)
    #Btn_Generte_Diagram.place(x=900,y=450,width=140,height=60)


    #Btn_Genrate_Decson = tkinter.Button(Decs,text="Generate Output",fg="black",command=Generate_Decsion_Output)
    #Btn_Genrate_Decson.place(x=155,y=400,width=140,height=60)

    lb_accur_score =   tkinter.Label(Cls,text="Accuracy Score:")
    lb_accur_score.place(x=150,y=150)

    lb_abs_error =   tkinter.Label(Cls,text="Mean Absolute error:")
    lb_abs_error.place(x=300,y=150)

    lb_mean_score =   tkinter.Label(Cls,text="RMSE:")
    lb_mean_score.place(x=450,y=150)

#    lb_rsqre =   tkinter.Label(Cls,text="Mean R-Square Error:")
#    lb_rsqre.place(x=600,y=150)

    Txt_accur  =   Text(Cls)
    Txt_accur.place(x=150,y=180,width=150,height=30)

    Txt_abs_error  =   Text(Cls)
    Txt_abs_error.place(x=300,y=180,width=150,height=30)

    Txt_mean_sqre  =   Text(Cls)
    Txt_mean_sqre.place(x=450,y=180,width=150,height=30)

#    Txt_rsqre  =   Text(Cls)
#    Txt_rsqre.place(x=600,y=180,width=150,height=30)

    lb_f1_score =   tkinter.Label(Cls,text="F1 Score:")
    lb_f1_score.place(x=750,y=150)

    lb_recall_score =   tkinter.Label(Cls,text="Recall Score:")
    lb_recall_score.place(x=860,y=150)

    lb_precson_score =   tkinter.Label(Cls,text="Precision Score:")
    lb_precson_score.place(x=1000,y=150)

    Txt_accur  =   Text(Cls)
    Txt_accur.place(x=150,y=180,width=150,height=30)

    Txt_abs_error  =   Text(Cls)
    Txt_abs_error.place(x=300,y=180,width=150,height=30)

    Txt_mean_sqre  =   Text(Cls)
    Txt_mean_sqre.place(x=450,y=180,width=150,height=30)

#    Txt_rsqre  =   Text(Cls)
#    Txt_rsqre.place(x=600,y=180,width=150,height=30)

    Txt_f1_score  =   Text(Cls)
    Txt_f1_score.place(x=740,y=180,width=150,height=30)

    Txt_recall_score  =   Text(Cls)
    Txt_recall_score.place(x=850,y=180,width=150,height=30)

    Txt_precson_score  =   Text(Cls)
    Txt_precson_score.place(x=980,y=180,width=150,height=30)

    lb_accur_score =   tkinter.Label(Cls,text="Confusion Matrix:",font=("Georgia", "18", "bold"), fg = "black")
    lb_accur_score.place(x=150,y= 250)
    
    Region_Unique = pd.unique(dataset['REGION'])
    print("The Unique Regions are ",Region_Unique)

   
    X = dataset.iloc[:, [0,1,2,3,5,6,7,9,10,11]].values
    y = dataset.iloc[:, 14].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=100)
  
    
   #Feature Scaling
  
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)


    model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p=2)
    fit = model.fit(X_train, y_train)
    print("The model fit for our training data",fit)
    y_predict = model.predict(X_test)



    acurcy_score =accuracy_score(y_test, y_predict)

    print("\n The accuracy score ",acurcy_score)
   
    

    print("The data for y_test is array for Region is ",y_test )

    print("The data for y_predict is array for Region is ",y_predict )

    print("Mean Absolute error is ",mean_absolute_error(y_test, y_predict ))
    
    print("The Rsquare error is ",r2_score(y_test, y_predict ))
    print("Mean Squared Error is ",mean_squared_error(y_test, y_predict))

   

    Txt_abs_error.insert(INSERT,"%.3f" % mean_absolute_error(y_test, y_predict )          )
    Txt_abs_error.tag_add("start", "1.0", END)
    Txt_abs_error.tag_config("start", font=("Georgia", "12", "bold"))

    Txt_mean_sqre.insert(INSERT,"%.3f" % mean_squared_error(y_test, y_predict)          )
    Txt_mean_sqre.tag_add("start", "1.0", END)
    Txt_mean_sqre.tag_config("start", font=("Georgia", "12", "bold"))


#    Txt_rsqre.insert(INSERT,"%.3f" % r2_score(y_test,y_predict          ))
#    Txt_rsqre.tag_add("start", "1.0", END)
#    Txt_rsqre.tag_config("start", font=("Georgia", "12", "bold"))

    


    confusionMatrix = confusion_matrix(y_test, y_predict)

    print("The confusionMatrix",confusionMatrix)
     # True Positive values
    lb_NE_TP =   tkinter.Label(Cls,text="TP" ,font=("Georgia", "14", "bold"), fg = "black")
    lb_NE_TP.place(x=200,y=310)
    
    lb_West_Regon =   tkinter.Label(Cls,text="Western" ,font=("Georgia", "10", "bold"), fg = "black")
    lb_West_Regon.place(x=50,y=360)
    
    lb_MidWest_Regon =   tkinter.Label(Cls,text="Mid Western" ,font=("Georgia", "10", "bold"), fg = "black")
    lb_MidWest_Regon.place(x=30,y=400)
    

    Txt_NE_TP  =   Text(Cls)
    Txt_NE_TP.place(x=150,y=340,width=100,height=50)
    
    Txt_NE_TP .insert(INSERT,       confusionMatrix[0, 0] )
    Txt_NE_TP .tag_add("start", "1.0", END)
    Txt_NE_TP.tag_config("start", font=("Georgia", "12", "bold"))
    
    #True Negative
    
    lb_NE_TN =   tkinter.Label(Cls,text="TN" ,font=("Georgia", "14", "bold"), fg = "black")
    lb_NE_TN.place(x=280,y=310)

    Txt_NE_TN  =   Text(Cls)
    Txt_NE_TN.place(x=250,y=340,width=100,height=50)
    
    Txt_NE_TN .insert(INSERT,       confusionMatrix[0, 1] )
    Txt_NE_TN .tag_add("start", "1.0", END)
    Txt_NE_TN.tag_config("start", font=("Georgia", "12", "bold"))
    
    #False Positive
    Txt_NE_FP  =   Text(Cls)
    Txt_NE_FP.place(x=150,y=390,width=100,height=50)
    
    lb_NE_FP =   tkinter.Label(Cls,text="FP" ,font=("Georgia", "14", "bold"), fg = "black")
    lb_NE_FP.place(x=180,y=440)
    
    Txt_NE_FP .insert(INSERT,       confusionMatrix[1, 0] )
    Txt_NE_FP .tag_add("start", "1.0", END)
    Txt_NE_FP.tag_config("start", font=("Georgia", "12", "bold"))
    
    # False Negative
    Txt_NE_FN  =   Text(Cls)
    Txt_NE_FN.place(x=250,y=390,width=100,height=50)
    
    lb_NE_FN =   tkinter.Label(Cls,text="FN" ,font=("Georgia", "14", "bold"), fg = "black")
    lb_NE_FN.place(x=280,y=440)
    
    
    
    
    Txt_NE_FN.insert(INSERT,       confusionMatrix[1, 1] )
    Txt_NE_FN.tag_add("start", "1.0", END)
    Txt_NE_FN.tag_config("start", font=("Georgia", "12", "bold"))
    
    TP= confusionMatrix[0, 0] 
    FN = confusionMatrix[1, 1]
    FP = confusionMatrix[1, 0]
    TN = confusionMatrix[0, 1]
    
    print("TP",TP)
    print("FN",FN)
    accur_score = (TP + FN) / 16904
   
    Txt_accur.insert(INSERT,"%.2f" %  accur_score)
    Txt_accur.tag_add("start", "1.0", END)
    Txt_accur.tag_config("start", font=("Georgia", "12", "bold"))
    
    print("\n The accuracy score ",accur_score)
    
    precson = TP /(TP + FP)
    print("Precision",precson )
    
    Txt_precson_score.insert(INSERT,"%.3f" % precson      )
    Txt_precson_score.tag_add("start", "1.0", END)
    Txt_precson_score.tag_config("start", font=("Georgia", "12", "bold"))
    
    recall = TP / (TP + FN)
    
    Txt_recall_score.insert(INSERT,"%.3f" % recall )
    Txt_recall_score.tag_add("start", "1.0", END)
    Txt_recall_score.tag_config("start", font=("Georgia", "12", "bold"))
    
    f1_score = 2 / ((1 / precson) + (1 /  recall))
    
    Txt_f1_score.insert(INSERT,"%.3f" % f1_score)
    Txt_f1_score.tag_add("start", "1.0", END)
    Txt_f1_score.tag_config("start", font=("Georgia", "12", "bold"))
    
   
   
#
#
#def Logistic_Regression():
#    
#    lgr = Tk()
#    lgr.title("Logistic Regression")
#    lb_Decs_headng =   tkinter.Label(lgr ,text="Logistic Regression",font=("Georgia", "26", "bold"), fg = "black")
#    lb_Decs_headng.place(x=450,y=60)
#
#    lb_accur_score =   tkinter.Label( lgr,text="Accuracy Score:")
#    lb_accur_score.place(x=150,y=150)
#
#    lb_abs_error =   tkinter.Label( lgr,text="Mean Absolute error:")
#    lb_abs_error.place(x=300,y=150)
#
#    lb_mean_score =   tkinter.Label( lgr,text="RMSE:")
#    lb_mean_score.place(x=450,y=150)
#
#    lb_rsqre =   tkinter.Label( lgr,text="Mean R-Square Error:")
#    lb_rsqre.place(x=600,y=150)
#
#    Txt_accur  =   Text( lgr)
#    Txt_accur.place(x=150,y=180,width=150,height=30)
#
#    Txt_abs_error  =   Text( lgr)
#    Txt_abs_error.place(x=300,y=180,width=150,height=30)
#
#    Txt_mean_sqre  =   Text( lgr)
#    Txt_mean_sqre.place(x=450,y=180,width=150,height=30)
#
#    Txt_rsqre  =   Text(lgr)
#    Txt_rsqre.place(x=600,y=180,width=150,height=30)
#
#    lb_f1_score =   tkinter.Label( lgr,text="F1 Score:")
#    lb_f1_score.place(x=750,y=150)
#
#    lb_recall_score =   tkinter.Label( lgr,text="Recall Score:")
#    lb_recall_score.place(x=860,y=150)
#
#    lb_precson_score =   tkinter.Label (lgr,text="Precision Score:")
#    lb_precson_score.place(x=1000,y=150)
#
#    Txt_accur  =   Text(lgr)
#    Txt_accur.place(x=150,y=180,width=150,height=30)
#
#    Txt_abs_error  =   Text(lgr)
#    Txt_abs_error.place(x=300,y=180,width=150,height=30)
#
#    Txt_mean_sqre  =   Text(lgr)
#    Txt_mean_sqre.place(x=450,y=180,width=150,height=30)
#
#    Txt_rsqre  =   Text(lgr)
#    Txt_rsqre.place(x=600,y=180,width=150,height=30)
#
#    Txt_f1_score  =   Text(lgr)
#    Txt_f1_score.place(x=740,y=180,width=150,height=30)
#
#    Txt_recall_score  =   Text(lgr)
#    Txt_recall_score.place(x=850,y=180,width=150,height=30)
#
#    Txt_precson_score  =   Text(lgr)
#    Txt_precson_score.place(x=980,y=180,width=150,height=30)
#
#    Txt_abs_error  =   Text(lgr)
#    Txt_abs_error.place(x=300,y=180,width=150,height=30)
#
#    
#    dataset = pd.read_csv("D:\Wildfire Reasons.csv")
#    
#
#    dataset= pd.DataFrame (dataset[['FIRE_SIZE','OWNER_CODE','CONT_DOY','LATITUDE','LONGITUDE','STAT_CAUSE_CODE']])
#    dataset.columns = dataset.columns.str.strip()
#    
#    X=dataset.drop('STAT_CAUSE_CODE',axis=1)
# 
#    print("X list ",X)
#
#    #print("Predictor Variables are ",predict_Var)
#    y= pd.DataFrame (dataset['STAT_CAUSE_CODE'])
#   
#   
#    scaler = Normalizer().fit(X)
#    normalizedX = scaler.transform(X)
#    X = pd.DataFrame(normalizedX)
#
#
#    logClassifier = linear_model.LogisticRegression(penalty='l2', tol=0.0001, C=100.0, random_state=1,max_iter=1000, n_jobs=-1)
#    X_train, X_test, y_train, y_test =   cross_validation.train_test_split(X,   y, test_size=0.30, random_state=100)
#
#    logclass = logClassifier.fit(X_train, y_train.values.ravel())
#    print("The log classifier is ",logclass)
#
#    predicted = logClassifier.predict(X_test)
#    print("Predicted  is ",predicted )
#    print("accuracy",accuracy_score(y_test, predicted))
#
#
#    print(classification_report(y_test, predicted))
#    print("\n\n")
#    print("F1 score",f1_score(y_test, predicted,average='weighted'))
#    print("Recall score",recall_score(y_test,predicted,average='weighted'))
#    print("Precision score",precision_score(y_test, predicted,average='weighted'))
#    print("Confusion Matrix:",confusion_matrix(y_test,predicted))
#    print("Root mean Square Error ",mean_squared_error(y_test,predicted))
#
#    
#    Txt_accur.insert(INSERT,"%.2f" % (accuracy_score(y_test, predicted)))
#    Txt_accur.tag_add("start", "1.0", END)
#    Txt_accur.tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_abs_error .insert(INSERT,"%.2f" % (mean_absolute_error(y_test,predicted ))         )
#    Txt_abs_error .tag_add("start", "1.0", END)
#    Txt_abs_error .tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_mean_sqre.insert(INSERT,"%.1f" % mean_squared_error(y_test,predicted)          )
#    Txt_mean_sqre.tag_add("start", "1.0", END)
#    Txt_mean_sqre.tag_config("start", font=("Georgia", "12", "bold"))
#
#
#    Txt_rsqre.insert(INSERT,"%.1f" % round(r2_score(y_test,predicted )          ))
#    Txt_rsqre.tag_add("start", "1.0", END)
#    Txt_rsqre.tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_f1_score.insert(INSERT,round(f1_score(y_test, predicted,average='weighted') ,2        ))
#    Txt_f1_score.tag_add("start", "1.0", END)
#    Txt_f1_score.tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_recall_score.insert(INSERT,round(recall_score(y_test, predicted,average='weighted'),2       ))
#    Txt_recall_score.tag_add("start", "1.0", END)
#    Txt_recall_score.tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_precson_score.insert(INSERT,round(precision_score(y_test, predicted,average='weighted'),2      ))
#    Txt_precson_score.tag_add("start", "1.0", END)
#    Txt_precson_score.tag_config("start", font=("Georgia", "12", "bold"))
#
#    lb_Light_hedng =   tkinter.Label(lgr,text="Lightining:",font=("Georgia", "12", "bold"), fg = "black")
#    lb_Light_hedng.place(x=330,y=320)
#
#    
#    lb_Debris_hedng =   tkinter.Label(lgr,text="Debris:",font=("Georgia", "12", "bold"), fg = "black")
#    lb_Debris_hedng.place(x=330,y=370)
#
#    confusionMatrix = confusion_matrix(y_test, predicted)
#    print("The confusion Matrix is:",confusionMatrix)
#    
#    lb_Light_TP =   tkinter.Label(lgr,text="TP")
#    lb_Light_TP.place(x=470,y=280)
#
#    Txt_Light_TP  =   Text(lgr)
#    Txt_Light_TP.place(x=450,y=300,width=100,height=50)
#
#    
#    Txt_Light_TP .insert(INSERT,       confusionMatrix[0, 0] )
#    Txt_Light_TP .tag_add("start", "1.0", END)
#    Txt_Light_TP.tag_config("start", font=("Georgia", "12", "bold"))
#
#    lb_NE_FP =   tkinter.Label(lgr,text="FP")
#    lb_NE_FP.place(x=580,y=280)
#
#    Txt_NE_FP  =   Text(lgr)
#    Txt_NE_FP.place(x=550,y=300,width=100,height=50)
#
#    Txt_NE_FP .insert(INSERT,       confusionMatrix[0, 1] )
#    Txt_NE_FP .tag_add("start", "1.0", END)
#    Txt_NE_FP.tag_config("start", font=("Georgia", "12", "bold"))
#
#    lb_NE_FN =   tkinter.Label(lgr,text="FN")
#    lb_NE_FN.place(x=470,y=400)
#
#    lb_NE_FN =   tkinter.Label(lgr,text="TN")
#    lb_NE_FN.place(x=600,y=400)
#
#
#    Txt_NE_FN  =   Text(lgr)
#    Txt_NE_FN.place(x=450,y=350,width=100,height=50)
#
#    Txt_NE_FN .insert(INSERT,       confusionMatrix[1, 0] )
#    Txt_NE_FN .tag_add("start", "1.0", END)
#    Txt_NE_FN.tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_NE_TN  =   Text(lgr)
#    Txt_NE_TN.place(x=550,y=350,width=100,height=50)
#
#    Txt_NE_TN .insert(INSERT,       confusionMatrix[1, 1] )
#    Txt_NE_TN .tag_add("start", "1.0", END)
#    Txt_NE_TN.tag_config("start", font=("Georgia", "12", "bold"))
#
#   
#
#    
#
#    

#def Random_Forest():
#
#    print("Random Forest")
#    lr = Tk()
#    lr.title("Random Forest")
#    lb_Decs_headng =   tkinter.Label(lr,text="Random Forest",font=("Georgia", "26", "bold"), fg = "black")
#    lb_Decs_headng.place(x=450,y=60)
#
#    dataset = pd.read_csv("D:\Wildfire Reasons.csv")
#
#
#    dataset= pd.DataFrame (dataset[['FIRE_SIZE','OWNER_CODE','CONT_DOY','LATITUDE','LONGITUDE','STAT_CAUSE_CODE']])
#    dataset.columns = dataset.columns.str.strip()
#    
#    X=dataset.drop('STAT_CAUSE_CODE',axis=1)
# 
#    print("X list ",X)
#
#    
#    y= pd.DataFrame (dataset['STAT_CAUSE_CODE'])
#    print("Predictor Variables are ",y)
#    
#
#   
#
#
#    # Split X and y into X_
#    X_train, X_test, y_train, y_test =train_test_split(X,   y, test_size=0.30, random_state=1)
#
#    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=100)
#    model = rf.fit(X_train, y_train.values.ravel())
#
#    predict = rf.predict(X_test)
#    accuracy = accuracy_score(y_test, predict)
#    print("The accuracy of Randome forest:",accuracy )
#
#    print(classification_report(y_test, predict))
#    print("\n\n")
#    print("F1 score",f1_score(y_test, predict,average='weighted'))
#    print("Recall score",recall_score(y_test,predict,average='weighted'))
#    print("Precision score",precision_score(y_test, predict,average='weighted'))
#    print("Confusion Matrix:",confusion_matrix(y_test,predict))
#    print("RMSE:",mean_squared_error(y_test, predict ))
#
#
#
#
#
#
#    lb_accur_score =   tkinter.Label( lr,text="Accuracy Score:")
#    lb_accur_score.place(x=150,y=150)
#
#    lb_abs_error =   tkinter.Label( lr,text="Mean Absolute error:")
#    lb_abs_error.place(x=300,y=150)
#
#    lb_mean_score =   tkinter.Label( lr,text="RMSE:")
#    lb_mean_score.place(x=480,y=150)
#
#    lb_rsqre =   tkinter.Label( lr,text="Mean R-Square Error:")
#    lb_rsqre.place(x=600,y=150)
#
#    Txt_accur  =   Text( lr)
#    Txt_accur.place(x=150,y=180,width=150,height=30)
#
#    Txt_abs_error  =   Text( lr)
#    Txt_abs_error.place(x=300,y=180,width=150,height=30)
#
#    Txt_mean_sqre  =   Text( lr)
#    Txt_mean_sqre.place(x=450,y=180,width=150,height=30)
#
#    Txt_rsqre  =   Text( lr)
#    Txt_rsqre.place(x=600,y=180,width=150,height=30)
#
#    lb_f1_score =   tkinter.Label( lr,text="F1 Score:")
#    lb_f1_score.place(x=750,y=150)
#
#    lb_recall_score =   tkinter.Label( lr,text="Recall Score:")
#    lb_recall_score.place(x=860,y=150)
#
#    lb_precson_score =   tkinter.Label ( lr,text="Precision Score:")
#    lb_precson_score.place(x=1000,y=150)
#
#    Txt_accur  =   Text(lr)
#    Txt_accur.place(x=150,y=180,width=150,height=30)
#
#    Txt_abs_error  =   Text(lr)
#    Txt_abs_error.place(x=300,y=180,width=150,height=30)
#
#    Txt_mean_sqre  =   Text(lr)
#    Txt_mean_sqre.place(x=450,y=180,width=150,height=30)
#
#    Txt_rsqre  =   Text(lr)
#    Txt_rsqre.place(x=600,y=180,width=150,height=30)
#
#    Txt_f1_score  =   Text(lr)
#    Txt_f1_score.place(x=740,y=180,width=150,height=30)
#
#    Txt_recall_score  =   Text(lr)
#    Txt_recall_score.place(x=850,y=180,width=150,height=30)
#
#    Txt_precson_score  =   Text(lr)
#    Txt_precson_score.place(x=980,y=180,width=150,height=30)
#
#
#
#    Txt_accur.insert(INSERT,"%.2f" % (accuracy_score(y_test, predict)))
#    Txt_accur.tag_add("start", "1.0", END)
#    Txt_accur.tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_abs_error .insert(INSERT,"%.1f" % (mean_absolute_error(y_test,predict ))         )
#    Txt_abs_error .tag_add("start", "1.0", END)
#    Txt_abs_error .tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_mean_sqre.insert(INSERT,"%.1f" % mean_squared_error(y_test, predict)          )
#    Txt_mean_sqre.tag_add("start", "1.0", END)
#    Txt_mean_sqre.tag_config("start", font=("Georgia", "12", "bold"))
#
#
#    Txt_rsqre.insert(INSERT,"%.1f" % round(r2_score(y_test,predict)          ))
#    Txt_rsqre.tag_add("start", "1.0", END)
#    Txt_rsqre.tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_f1_score.insert(INSERT,round(f1_score(y_test, predict,average='weighted') ,2        ))
#    Txt_f1_score.tag_add("start", "1.0", END)
#    Txt_f1_score.tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_recall_score.insert(INSERT,round(recall_score(y_test, predict,average='weighted'),2       ))
#    Txt_recall_score.tag_add("start", "1.0", END)
#    Txt_recall_score.tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_precson_score.insert(INSERT,round(precision_score(y_test, predict,average='weighted'),2      ))
#    Txt_precson_score.tag_add("start", "1.0", END)
#    Txt_precson_score.tag_config("start", font=("Georgia", "12", "bold"))
#
#    lb_Light_hedng =   tkinter.Label(lr,text="Lightining:",font=("Georgia", "12", "bold"), fg = "black")
#    lb_Light_hedng.place(x=330,y=320)
#
#    
#    lb_Debris_hedng =   tkinter.Label(lr,text="Debris:",font=("Georgia", "12", "bold"), fg = "black")
#    lb_Debris_hedng.place(x=330,y=370)
#
#    confusionMatrix = confusion_matrix(y_test, predict)
#    print("The confusion Matrix is:",confusionMatrix)
#    
#    lb_Light_TP =   tkinter.Label(lr,text="TP")
#    lb_Light_TP.place(x=470,y=280)
#
#    Txt_Light_TP  =   Text(lr)
#    Txt_Light_TP.place(x=450,y=300,width=100,height=50)
#
#    
#    Txt_Light_TP .insert(INSERT,       confusionMatrix[0, 0] )
#    Txt_Light_TP .tag_add("start", "1.0", END)
#    Txt_Light_TP.tag_config("start", font=("Georgia", "12", "bold"))
#
#    lb_NE_FP =   tkinter.Label(lr,text="FP")
#    lb_NE_FP.place(x=580,y=280)
#
#    Txt_NE_FP  =   Text(lr)
#    Txt_NE_FP.place(x=550,y=300,width=100,height=50)
#
#    Txt_NE_FP .insert(INSERT,       confusionMatrix[0, 1] )
#    Txt_NE_FP .tag_add("start", "1.0", END)
#    Txt_NE_FP.tag_config("start", font=("Georgia", "12", "bold"))
#
#    lb_NE_FN =   tkinter.Label(lr,text="FN")
#    lb_NE_FN.place(x=470,y=400)
#
#    lb_NE_FN =   tkinter.Label(lr,text="TN")
#    lb_NE_FN.place(x=600,y=400)
#
#
#    Txt_NE_FN  =   Text(lr)
#    Txt_NE_FN.place(x=450,y=350,width=100,height=50)
#
#    Txt_NE_FN .insert(INSERT,       confusionMatrix[1, 0] )
#    Txt_NE_FN .tag_add("start", "1.0", END)
#    Txt_NE_FN.tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_NE_TN  =   Text(lr)
#    Txt_NE_TN.place(x=550,y=350,width=100,height=50)
#
#    Txt_NE_TN .insert(INSERT,       confusionMatrix[1, 1] )
#    Txt_NE_TN .tag_add("start", "1.0", END)
#    Txt_NE_TN.tag_config("start", font=("Georgia", "12", "bold"))
#    
#
#    

def Model_Comparsion():
    
    print("Model Comparsion")
    Mcp = Tk()
    Mcp.title("Model Comparsion")
    lb_Decs_headng =   tkinter.Label(Mcp,text="Model Comparsion",font=("Georgia", "26", "bold"), fg = "black")
    lb_Decs_headng.place(x=450,y=60)

    lb_Decsion_Tree =   tkinter.Label( Mcp,text="Decision Tree" ,font=("Georgia", "14", "bold"), fg = "black")
    lb_Decsion_Tree.place(x=150,y=200)

    lb_Classfon =   tkinter.Label( Mcp,text="Classification" ,font=("Georgia", "14", "bold"), fg = "black")
    lb_Classfon.place(x=150,y=230)

#    lb_Logistc_Regr =   tkinter.Label( Mcp,text="Logistic Regression" ,font=("Georgia", "14", "bold"), fg = "black")
#    lb_Logistc_Regr.place(x=120,y=260)
#
#    lb_Suprt_Vector=   tkinter.Label( Mcp,text="Random Forest" ,font=("Georgia", "14", "bold"), fg = "black")
#    lb_Suprt_Vector.place(x=120,y=290)

    Txt_Decs_Accur =   Text(Mcp)
    Txt_Decs_Accur.place(x=320,y=200,width=150,height=30)

    Txt_Classfiction =   Text(Mcp)
    Txt_Classfiction.place(x=320,y=230,width=150,height=30)

#    Txt_Logistc_Regr =   Text(Mcp)
#    Txt_Logistc_Regr.place(x=320,y=260,width=150,height=30)
    
#    Txt_Random_frst =   Text(Mcp)
#    Txt_Random_frst.place(x=320,y=290,width=150,height=30)

    lb_Accurcy =   tkinter.Label( Mcp,text="Accuracy" ,font=("Georgia", "14", "bold"), fg = "black")
    lb_Accurcy.place(x=320,y=170)

    Txt_Decs_Mean_abs =   Text(Mcp)
    Txt_Decs_Mean_abs.place(x=450,y=200,width=150,height=30)

    Txt_Classfiction_Mean_abs =   Text(Mcp)
    Txt_Classfiction_Mean_abs.place(x=450,y=230,width=150,height=30)

#    Txt_Logistc_Regr_Mean_abs =   Text(Mcp)
#    Txt_Logistc_Regr_Mean_abs.place(x=450,y=260,width=150,height=30)
#    
#    Txt_Random_frst_Mean_abs =   Text(Mcp)
#    Txt_Random_frst_Mean_abs.place(x=450,y=290,width=150,height=30)

    lb_Accurcy =   tkinter.Label( Mcp,text="RMSE" ,font=("Georgia", "14", "bold"), fg = "black")
    lb_Accurcy.place(x=450,y=170)


    Txt_Decs_Accur.insert(INSERT,0.79      )
    Txt_Decs_Accur.tag_add("start", "1.0", END)
    Txt_Decs_Accur.tag_config("start", font=("Georgia", "12", "bold"))

    Txt_Classfiction.insert(INSERT,0.78      )
    Txt_Classfiction.tag_add("start", "1.0", END)
    Txt_Classfiction.tag_config("start", font=("Georgia", "12", "bold"))

#    Txt_Logistc_Regr.insert(INSERT,0.82     )
#    Txt_Logistc_Regr.tag_add("start", "1.0", END)
#    Txt_Logistc_Regr.tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_Random_frst.insert(INSERT,0.92)
#    Txt_Random_frst.tag_add("start", "1.0", END)
#    Txt_Random_frst.tag_config("start", font=("Georgia", "12", "bold"))

    Txt_Decs_Mean_abs.insert(INSERT,0.208   )
    Txt_Decs_Mean_abs.tag_add("start", "1.0", END)
    Txt_Decs_Mean_abs.tag_config("start", font=("Georgia", "12", "bold"))

    Txt_Classfiction_Mean_abs.insert(INSERT,0.220     )
    Txt_Classfiction_Mean_abs.tag_add("start", "1.0", END)
    Txt_Classfiction_Mean_abs.tag_config("start", font=("Georgia", "12", "bold"))

#    Txt_Logistc_Regr_Mean_abs.insert(INSERT,2.9      )
#    Txt_Logistc_Regr_Mean_abs.tag_add("start", "1.0", END)
#    Txt_Logistc_Regr_Mean_abs.tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_Random_frst_Mean_abs.insert(INSERT,1.3     )
#    Txt_Random_frst_Mean_abs.tag_add("start", "1.0", END)
#    Txt_Random_frst_Mean_abs.tag_config("start", font=("Georgia", "12", "bold"))


    Txt_Decs_precson =   Text(Mcp)
    Txt_Decs_precson.place(x=600,y=200,width=150,height=30)

    Txt_Classfiction_precson =   Text(Mcp)
    Txt_Classfiction_precson.place(x=600,y=230,width=150,height=30)

#    Txt_Logistc_Regr_precson  =   Text(Mcp)
#    Txt_Logistc_Regr_precson.place(x=600,y=260,width=150,height=30)
#
#    Txt_Random_frst_precson =   Text(Mcp)
#    Txt_Random_frst_precson.place(x=600,y=290,width=150,height=30)

    lb_precson=   tkinter.Label( Mcp,text="Precision" ,font=("Georgia", "14", "bold"), fg = "black")
    lb_precson.place(x=610,y=170)

    Txt_Decs_recall =   Text(Mcp)
    Txt_Decs_recall.place(x=750,y=200,width=150,height=30)

    Txt_Classfiction_recall=   Text(Mcp)
    Txt_Classfiction_recall.place(x=750,y=230,width=150,height=30)

#    Txt_Logistc_Regr_recall  =   Text(Mcp)
#    Txt_Logistc_Regr_recall.place(x=750,y=260,width=150,height=30)
#
#    Txt_Random_frst_recall =   Text(Mcp)
#    Txt_Random_frst_recall.place(x=750,y=290,width=150,height=30)


    Txt_Decs_precson.insert(INSERT,0.822      )
    Txt_Decs_precson.tag_add("start", "1.0", END)
    Txt_Decs_precson.tag_config("start", font=("Georgia", "12", "bold"))


    Txt_Classfiction_precson.insert(INSERT,0.820      )
    Txt_Classfiction_precson.tag_add("start", "1.0", END)
    Txt_Classfiction_precson.tag_config("start", font=("Georgia", "12", "bold"))


#    Txt_Logistc_Regr_precson .insert(INSERT,0.81    )
#    Txt_Logistc_Regr_precson .tag_add("start", "1.0", END)
#    Txt_Logistc_Regr_precson .tag_config("start", font=("Georgia", "12", "bold"))
#
#
#    Txt_Random_frst_precson.insert(INSERT,0.92    )
#    Txt_Random_frst_precson.tag_add("start", "1.0", END)
#    Txt_Random_frst_precson.tag_config("start", font=("Georgia", "12", "bold"))





    lb_precson=   tkinter.Label( Mcp,text="Recall" ,font=("Georgia", "14", "bold"), fg = "black")
    lb_precson.place(x=750,y=170)


    Txt_Decs_recall.insert(INSERT,0.686     )
    Txt_Decs_recall.tag_add("start", "1.0", END)
    Txt_Decs_recall.tag_config("start", font=("Georgia", "12", "bold"))

    Txt_Classfiction_recall.insert(INSERT,0.681      )
    Txt_Classfiction_recall.tag_add("start", "1.0", END)
    Txt_Classfiction_recall.tag_config("start", font=("Georgia", "12", "bold"))

#    Txt_Logistc_Regr_recall.insert(INSERT,0.82     )
#    Txt_Logistc_Regr_recall.tag_add("start", "1.0", END)
#    Txt_Logistc_Regr_recall.tag_config("start", font=("Georgia", "12", "bold"))
#
#    Txt_Random_frst_recall.insert(INSERT,0.92    )
#    Txt_Random_frst_recall.tag_add("start", "1.0", END)
#    Txt_Random_frst_recall.tag_config("start", font=("Georgia", "12", "bold"))

    



    

def Generte_Diagram():
    
    
    dataset = pd.read_csv("D:\Copy of Modified Wildfire Dataset for Analysis.csv")
    print(1)

    dataset= pd.DataFrame (dataset[['FIRE_YEAR','DISCOVERY_DOY','DISCOVERY_TIME', 'STAT_CAUSE_CODE','CONT_DOY','CONT_TIME',
                                      'FIRE_SIZE','LATITUDE','LONGITUDE' ,'REGION']])
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
#Count of Fire Yearly
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
    wildFire_year = { 'US_Regions':['Midwest','Northeast','West','South'],
                      '1992-1999' :[26824,8288,6141,30344],
                      '2000-2008' :[26546,26033,40940,28239] 
                    
                    

                                               }

    
    #'Children': [2807]
    df = pd.DataFrame(wildFire_year, columns = ['US_Regions', '1992-1999',  '2000-2008' ])

    # Setting the positions and width for the bars
    pos = list(range(len(df['1992-1999']))) 
    width = 0.25 

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10,5))

    # Create a bar with pre_score data,
    # in position pos,
    plt.bar(pos ,
        #using df['pre_score'] data,
        df['1992-1999'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#A9A9A9', 
        # with label the first value in first_name
        label=df['US_Regions'])

    # Create a bar with mid_score data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos], 
        #using df['mid_score'] data,
        df['2000-2008'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#A52A2A', 
        # with label the second value in first_name
        label=df['US_Regions'])


   
   
    
   
    # Set the y axis label
    ax.set_ylabel('Count of Wild Fire Incidents (Yearly Basis) ')
    ax.set_xlabel('US Regions')
    
    # Set the chart's title
    ax.set_title('Wild Fire Incidents on Yearly Basis ')

    # Set the position of the x ticks
    ax.set_xticks([p +  0.5* width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(df['US_Regions'])

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+width * 5)
    plt.ylim([0, max(df['1992-1999'] + df['2000-2008'] + 4000)] )
             
   # Adding the legend and showing the plot
    plt.legend([ '1992-1999' ,'2000-2008'], loc='upper left')
    plt.grid()
    plt.show()


    


     
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
#Category of Fire Size
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    reson_wildfre = { 'US_Regions':['Midwest','Northeast','West','South'],
                      'A' :[15550,11702,15516,10488],
                      'B' :[29042,18927,21240,32445] ,
                      'C' : [7518,6035,7555,13709]
                    

                                               }

    
    #'Children': [2807]
    df = pd.DataFrame(reson_wildfre, columns = ['US_Regions', 'A', 'B','C' ])

    # Setting the positions and width for the bars
    pos = list(range(len(df['A']))) 
    width = 0.1 

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10,5))

    # Create a bar with pre_score data,
    # in position pos,
    plt.bar(pos ,
        #using df['pre_score'] data,
        df['A'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#A9A9A9', 
        # with label the first value in first_name
        label=df['US_Regions'])

    # Create a bar with mid_score data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos], 
        #using df['mid_score'] data,
        df['B'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#A52A2A', 
        # with label the second value in first_name
        label=df['US_Regions'])


    plt.bar([p + width*2 for p in pos], 
        #using df['mid_score'] data,
        df['C'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color=  '#00008B', 
        # with label the second value in first_name
        label=df['US_Regions'])
   
   
    
   
    # Set the y axis label
    ax.set_ylabel('No of Category of Fire')
    ax.set_xlabel('US Regions')
    
    # Set the chart's title
    ax.set_title('Category of Fire')

    # Set the position of the x ticks
    ax.set_xticks([p + 1 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(df['US_Regions'])

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+width*5)
    plt.ylim([0, max(df[ 'A'] + df['B'] + df['C'] + 4000 )] )
             
   # Adding the legend and showing the plot
    plt.legend([ 'A: [0-0.25]  Acres' ,'B: [0.26-9.9]  Acres','C: [10-99.9]  Acres'], loc='upper left')
    plt.grid()
    plt.show()

    
#----------------------------------------------------------------------------------------------------------------------------------------------------   
#No of Wild Fire Incidents  
#-------------------------------------------------------------------------------------------------------------------------------------------------------

    reson_wildfre = { 'US_Regions':['Midwest','Northeast','West','South'],
                      'Arson' :[9505,7058,7472,18779],
                      'Campfire' :[941,1251,2563,783] ,
                      'Children': [2807,1300,776,887],
                      'Debris_Running':[9724,8326,9565,17371],
                      'Equipment Use' : [3264,2047,3723,3481],
                      'Lightning':      [1754,6051,7552,1222],
                      'Miscellaneous':  [5733,4665,9089,7715],
                      'Missing /Undefined': [18000,6166,6544,4333]

                                               }

    
    #'Children': [2807]
    df = pd.DataFrame(reson_wildfre, columns = ['US_Regions', 'Arson', 'Campfire', 'Children','Debris_Running', 'Equipment Use' , 'FireWorks', 'Lightning',
                                                'Miscellaneous', 'Missing /Undefined' ])
    
    # Setting the positions and width for the bars
    pos = list(range(len(df['Arson']))) 
    width = 0.1 

    # Plotting the bars
    fig, ax = plt.subplots(figsize=(10,5))

    # Create a bar with pre_score data,
    # in position pos,
    plt.bar(pos ,
        #using df['pre_score'] data,
        df['Arson'], 
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#A9A9A9', 
        # with label the first value in first_name
        label=df['US_Regions'])

    # Create a bar with mid_score data,
    # in position pos + some width buffer,
    plt.bar([p + width for p in pos], 
        #using df['mid_score'] data,
        df['Campfire'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#A52A2A', 
        # with label the second value in first_name
        label=df['US_Regions'])

    plt.bar([p + width*2 for p in pos], 
        #using df['mid_score'] data,
        df['Children'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color='#006400', 
        # with label the second value in first_name
        label=df['US_Regions'])

    plt.bar([p + width*3 for p in pos], 
        #using df['mid_score'] data,
        df['Debris_Running'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color= '#008B8B', 
        # with label the second value in first_name
        label=df['US_Regions'])

    
    plt.bar([p + width*4 for p in pos], 
        #using df['mid_score'] data,
        df['Equipment Use'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color=  '#FF00FF', 
        # with label the second value in first_name
        label=df['US_Regions'])



    plt.bar([p + width*5 for p in pos], 
        #using df['mid_score'] data,
        df['Lightning'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color=   '#FFFF00', 
        # with label the second value in first_name
        label=df['US_Regions'])


    plt.bar([p + width*6 for p in pos], 
        #using df['mid_score'] data,
        df['Miscellaneous'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color=   '#FFC0CB', 
        # with label the second value in first_name
        label=df['US_Regions'])


    plt.bar([p + width*7 for p in pos], 
        #using df['mid_score'] data,
        df['Missing /Undefined'],
        # of width
        width, 
        # with alpha 0.5
        alpha=0.5, 
        # with color
        color=   '#000000', 
        # with label the second value in first_name
        label=df['US_Regions'])



  
    
   
    # Set the y axis label
    ax.set_ylabel('No of WildFire Incidents')
    ax.set_xlabel('US Regions')
    
    # Set the chart's title
    ax.set_title('WildFire Reasons')

    # Set the position of the x ticks
    ax.set_xticks([p + 4 * width for p in pos])

    # Set the labels for the x ticks
    ax.set_xticklabels(df['US_Regions'])

    # Setting the x-axis and y-axis limits
    plt.xlim(min(pos)-width, max(pos)+width*5)
    plt.ylim([0, min(df[ 'Arson'] + df['Campfire'] +df['Children'] + df['Debris_Running'] + df['Equipment Use'] + df['Lightning']
                      + df['Miscellaneous'] + df['Missing /Undefined']  -1000 )] )
             
   # Adding the legend and showing the plot
    plt.legend([ 'Arson','Campfire', 'Children','Debris_Running','Equipment Use','Lightning' ,'Miscellaneous','Missing /Undefined'], loc='upper left')
    plt.grid()
    plt.show()

    

#-------------------------------------------------------------------------------------------------------------------------------------------------------

   
    fig, axes = plt.subplots()
    ax= np.array([0,1,2,3])

    
    ay = np.array([9770,28275  ,10821,54779])
    
    us_regions = ['Midwest','Northeast','South','West']


    plt.xticks(ax, us_regions)
    plt.bar(ax,ay, label="US Wild Fires", color='g')
    plt.xlabel('US Regions')
    plt.ylabel('Total Area(Square km)')
    plt.title("US Wild Fires")
    plt.show()
    
   
    us_regions = ['Midwest','Northeast','South','West']

    ay_long = [82.13 ,94.63 ,89.40 ,93.79]
    plt.xticks(ax, us_regions)
    plt.bar(ax,ay_long, label="US Wild Fires", color='g')
    plt.xlabel('US Regions')
    plt.ylabel('Average Longitutde(Wild Fire Occured)')
    plt.title("US Wild Fires Direction")
    #plt.subplot(4,4,1)
    plt.show()
            
#------------------------------------------------------------------------------------------------------------------------------
    
    
  
    
root = Tk()

root.title("Wild Fire Dataset Prediction")

menu = Menu(root)
root.config(menu=menu)
filemenu = Menu(menu)

menu.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="Load Dataset", command=Load_Dataset)
#filemenu.add_command(label="Feature Selection", command=Feature_Selection)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)

Alogrthm_menu = Menu(menu)
menu.add_cascade(label="Technique", menu=Alogrthm_menu )
Alogrthm_menu.add_command(label="Decision Tree", command=Decision_Tree)
Alogrthm_menu.add_command(label="Classification", command=Classification)
#Alogrthm_menu.add_command(label="Logistic Regression", command=Logistic_Regression)
#Alogrthm_menu.add_command(label="Support Vector Machine", command=Support_Vector_Machine)
#Alogrthm_menu.add_command(label="Random Forest", command=Random_Forest)



Chartmenu = Menu(menu)
menu.add_cascade(label="Charts", menu=Chartmenu )
Chartmenu.add_command(label="Bar Chart", command= Generte_Diagram)

Modelmenu = Menu(menu)
menu.add_cascade(label="Model Comparsion", menu=Modelmenu )
Modelmenu.add_command(label="Comparison", command= Model_Comparsion)

root.mainloop()
