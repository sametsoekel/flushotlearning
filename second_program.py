import sys
import os
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QApplication,QDialog,QMessageBox,QFileDialog
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5 import QtGui
import pandas as pd
from catboost import CatBoostClassifier

h1n1_predictor=CatBoostClassifier().load_model('C:\\Users\\Public\\h1n1_predictor.cbm')
seasonal_predictor=CatBoostClassifier().load_model('C:\\Users\\Public\\seasonal_predictor.cbm')


def machine(dataset):
    ms=pd.get_dummies(dataset[['sex','age_group','education','race','income_poverty','marital_status','rent_or_own','employment_status','census_msa']])
    mydummies=ms[['sex_Female','age_group_18 - 34 Years','age_group_35 - 44 Years','age_group_45 - 54 Years','age_group_55 - 64 Years','education_12 Years','education_< 12 Years',
'education_College Graduate','race_Black','race_Hispanic','race_Other or Multiple','income_poverty_<= $75,000, Above Poverty','income_poverty_> $75,000','marital_status_Married',
             'rent_or_own_Own','employment_status_Employed','employment_status_Not in Labor Force','census_msa_MSA, Not Principle  City','census_msa_MSA, Principle City' ]]
    dataset=dataset.drop(['sex','age_group','education','race','income_poverty','marital_status','rent_or_own','employment_status','census_msa'],axis=1)
    dataset = pd.concat([dataset.reset_index(drop=True), mydummies], axis=1)
    dataset=dataset.dropna()
    dataset=dataset.drop(['respondent_id','hhs_geo_region','employment_industry','employment_occupation'],axis=1)
    dataset[['sex_Female','age_group_18 - 34 Years','age_group_35 - 44 Years','age_group_45 - 54 Years','age_group_55 - 64 Years','education_12 Years','education_< 12 Years','education_College Graduate','race_Black','race_Hispanic','race_Other or Multiple','income_poverty_<= $75,000, Above Poverty','income_poverty_> $75,000','marital_status_Married','rent_or_own_Own','employment_status_Employed','census_msa_MSA, Not Principle  City','census_msa_MSA, Principle City','doctor_recc_seasonal']]=dataset[['sex_Female','age_group_18 - 34 Years','age_group_35 - 44 Years','age_group_45 - 54 Years','age_group_55 - 64 Years','education_12 Years','education_< 12 Years','education_College Graduate','race_Black','race_Hispanic','race_Other or Multiple','income_poverty_<= $75,000, Above Poverty','income_poverty_> $75,000','marital_status_Married','rent_or_own_Own','employment_status_Employed','census_msa_MSA, Not Principle  City','census_msa_MSA, Principle City','doctor_recc_seasonal']].astype('int64')
    dataset[["behavioral_antiviral_meds","behavioral_avoidance","behavioral_face_mask","behavioral_wash_hands","behavioral_large_gatherings","behavioral_outside_home","behavioral_touch_face","doctor_recc_h1n1","doctor_recc_seasonal",'doctor_recc_seasonal','chronic_med_condition','child_under_6_months','health_worker','health_insurance']]=dataset[["behavioral_antiviral_meds","behavioral_avoidance","behavioral_face_mask","behavioral_wash_hands","behavioral_large_gatherings","behavioral_outside_home","behavioral_touch_face","doctor_recc_h1n1","doctor_recc_seasonal",'doctor_recc_seasonal','chronic_med_condition','child_under_6_months','health_worker','health_insurance']].astype('int64')
    dataset[['h1n1_concern','h1n1_knowledge','opinion_h1n1_vacc_effective','opinion_h1n1_risk','opinion_h1n1_sick_from_vacc','opinion_seas_vacc_effective','opinion_seas_risk','opinion_seas_sick_from_vacc','household_adults','household_children']]=dataset[['h1n1_concern','h1n1_knowledge','opinion_h1n1_vacc_effective','opinion_h1n1_risk','opinion_h1n1_sick_from_vacc','opinion_seas_vacc_effective','opinion_seas_risk','opinion_seas_sick_from_vacc','household_adults','household_children']].astype('int64')
    return dataset.drop(['phone'],axis=1)
    
            
class Pencere(QDialog):
    
    def __init__(self):
        
        super(Pencere,self).__init__()
        loadUi("C:\\Users\\Public\\second_interface.ui",self)
        self.pushButton.clicked.connect(self.selectFile)
        self.pushButton_2.clicked.connect(self.justh1n1)
        self.pushButton_3.clicked.connect(self.justseasonal)
        self.pushButton_4.clicked.connect(self.both)
        self.pushButton_5.clicked.connect(self.onSave)
        self.pushButton_6.clicked.connect(self.SMS)
    
        
    def selectFile(self):
        self.lineEdit.setText(QFileDialog.getOpenFileName()[0])
        
    def justh1n1(self):
        try:
            self.test=SelectingFuncs(None,pd.read_csv(self.lineEdit.text().replace('\\','/')),self.listWidget)
            self.test.justh1n1()
        except FileNotFoundError:
            pass
        
    def justseasonal(self):
        try:
            self.test=SelectingFuncs(None,pd.read_csv(self.lineEdit.text().replace('\\','/')),self.listWidget)
            self.test.justseasonal()
        except FileNotFoundError:
            pass
        
    def both(self):
        try:
            self.test=SelectingFuncs(None,pd.read_csv(self.lineEdit.text().replace('\\','/')),self.listWidget)
            self.test.both()
        except FileNotFoundError:
            pass
        
    def onSave(self):
        self.saveFile(True)
    def saveFile(self, showDialog):
        savePath = os.path.join(os.getcwd(),
                                'ListOfPhoneNumbers')

        if showDialog:
            savePath = QFileDialog.getSaveFileName(self,
                                                         'Save text file',
                                                         savePath,
                                                         '*.txt')


        if len(savePath) > 0:
            with open(savePath[0], 'w') as theFile:
                for i in range(self.listWidget.count()):
                    theFile.write(''.join([str(self.listWidget.item(i).text()),
                                           '\n']))
                                           
    def SMS(self):
        buttonReply = QMessageBox.information(self, 'Information', "Message has sent all numbers")
        self.show()
        
#########################################################
##Selecting Functions are controlled by a QThread object.
#########################################################      
class SelectingFuncs(QThread):
    def __init__(self,parent=None,hamdataset='0',liste='0'):
        super(SelectingFuncs,self).__init__(parent)
        self.hamdataset=hamdataset
        self.liste=liste
        
    def justh1n1(self):
            array = h1n1_predictor.predict(machine(self.hamdataset))
            for i,j in zip(machine(self.hamdataset).index,range(int(len(machine(self.hamdataset).index-1)))):
                if array[j]==0:
                    self.liste.addItem(str(self.hamdataset['phone'][i]))
        
                
    def justseasonal(self):
            array=seasonal_predictor.predict(machine(self.hamdataset))
            for i,j in zip(machine(self.hamdataset).index,range(len(machine(self.hamdataset).index-1))):
                if array[j]==0:
                    self.liste.addItem(str(self.hamdataset['phone'][i]))

    
    def both(self):
            array1=seasonal_predictor.predict(machine(self.hamdataset))
            array2=h1n1_predictor.predict(machine(self.hamdataset))
            for i,j in zip(machine(self.hamdataset).index,range(len(machine(self.hamdataset).index-1))):
                if array1[j]==0 and array2[j]==0:
                    self.liste.addItem(str(self.hamdataset['phone'][i]))

        
if __name__ == '__main__':
    app=QApplication(sys.argv)
    widget=Pencere()
    widget.show()
    app.exit(app.exec())