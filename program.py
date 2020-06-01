#####model files and interface should be in c:/users/public##############



import sys
from PyQt5.QtCore import pyqtSlot,QUrl,QTimer,QTime,QThread,pyqtSignal
from PyQt5.QtWidgets import QApplication,QDialog,QMessageBox
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5 import QtGui
import numpy as np
from catboost import CatBoostClassifier

class Pencere(QDialog):
    
    def __init__(self):
        
        super(Pencere,self).__init__()
        loadUi("C:\\Users\\Public\\interface.ui",self)
        
        self.pushButton.clicked.connect(self.yaz)
        self.model1=CatBoostClassifier()
        self.model2=CatBoostClassifier()
        self.model1.load_model('C:\\Users\\Public\\h1n1_predictor.cbm')
        self.model2.load_model('C:\\Users\\Public\\seasonal_predictor.cbm')
    def yaz(self):
        h1n1_concern=int(self.comboBox_21.currentText())
        h1n1_knowledge=int(self.comboBox_21.currentText())
        beh_ant_med=int(self.comboBox_34.currentIndex())
        beh_avoid=int(self.comboBox_35.currentIndex())
        beh_mask=int(self.comboBox_36.currentIndex())
        beh_hands=int(self.comboBox_37.currentIndex())
        beh_gather=int(self.comboBox_39.currentIndex())
        beh_out=int(self.comboBox_38.currentIndex())
        beh_touch=int(self.comboBox_43.currentIndex())
        recc_h1n1=int(self.comboBox_42.currentIndex())
        recc_season=int(self.comboBox_41.currentIndex())
        chronic=int(self.comboBox_40.currentIndex())
        undersix=int(self.comboBox_9.currentIndex())
        health_worker=int(self.comboBox_14.currentIndex())
        insurance=int(self.comboBox_10.currentIndex())
        op_h1ef=int(self.comboBox_15.currentText())
        op_h1risk=int(self.comboBox_16.currentIndex())
        op_h1sick=int(self.comboBox_17.currentIndex())
        op_seef=int(self.comboBox_18.currentText())
        op_serisk=int(self.comboBox_19.currentIndex())
        op_sesick=int(self.comboBox_20.currentIndex())
        if self.comboBox_2.currentText()=='Female':
            sex=1
        else:
            sex=0
        if self.comboBox.currentText()=='18 - 34 Years':
            birincigrupyas=1
            ikincigrupyas=0
            ucuncugrupyas=0
            dorduncugrupyas=0
        elif self.comboBox.currentText()=='35 - 44 Years':
            birincigrupyas=0
            ikincigrupyas=1
            ucuncugrupyas=0
            dorduncugrupyas=0
        elif self.comboBox.currentText()=='45 - 54 Years':
            birincigrupyas=0
            ikincigrupyas=0
            ucuncugrupyas=1
            dorduncugrupyas=0
        elif self.comboBox.currentText()=='55 - 64 Years':
            birincigrupyas=0
            ikincigrupyas=0
            ucuncugrupyas=0
            dorduncugrupyas=1
        else:
            birincigrupyas=0
            ikincigrupyas=0
            ucuncugrupyas=0
            dorduncugrupyas=0
        if self.comboBox_3.currentText()=='12 Years':
            birincigrupegitim=1
            ikincigrupegitim=0
            ucuncugrupegitim=0
        elif self.comboBox_3.currentText()=='< 12 Years':
            birincigrupegitim=0
            ikincigrupegitim=1
            ucuncugrupegitim=0
        elif self.comboBox_3.currentText()=='College Graduate':
            birincigrupegitim=0
            ikincigrupegitim=0
            ucuncugrupegitim=1
        else:
            birincigrupegitim=0
            ikincigrupegitim=0
            ucuncugrupegitim=0
        if self.comboBox_4.currentText()=='Black':
            birincigrupirk=1
            ikincigrupirk=0
            ucuncugrupirk=0
        elif self.comboBox_4.currentText()=='Hispanic':
            birincigrupirk=0
            ikincigrupirk=1
            ucuncugrupirk=0
        elif self.comboBox_4.currentText()=='Other or Multiple':
            birincigrupirk=0
            ikincigrupirk=0
            ucuncugrupirk=1
        else:
            birincigrupirk=0
            ikincigrupirk=0
            ucuncugrupirk=0
        if self.comboBox_8.currentText()=='Less than 75.000$':
            birincigelir=1
            ikincigelir=0
        elif self.comboBox_8.currentText()=='More than 75.000$':
            birincigelir=0
            ikincigelir=1
        else:
            birincigelir=0
            ikincigelir=0
        if self.comboBox_6.currentText()=='Employed':
            employbir=1
            employiki=0
        elif self.comboBox_6.currentText()=='Unemployed':
            employbir=0
            employiki=1
        else:
            employbir=0
            employiki=0
            
        if self.comboBox_11.currentText()=='City':
            censusbir=1
            censusiki=0
        elif self.comboBox_11.currentText()=='Metropol':
            censusbir=0
            censusiki=1
        else:
            censusbir=0
            censusiki=0
        if self.comboBox_5.currentText()=='Married':
            marital=1
        else:
            marital=0
        if self.comboBox_7.currentText()=='Married':
            own=1
        else:
            own=0
        hhadult=int(self.comboBox_12.currentText())
        hhchild=int(self.comboBox_13.currentText())
        
        
        dizi=np.array([h1n1_concern,h1n1_knowledge,beh_ant_med,beh_avoid,beh_mask,beh_hands,beh_gather,beh_out,beh_touch,recc_h1n1,recc_season,chronic,undersix,health_worker,insurance,op_h1ef,op_h1risk,op_h1sick,op_seef,op_serisk,op_sesick,hhadult,hhchild,sex,birincigrupyas,ikincigrupyas,ucuncugrupyas,dorduncugrupyas,birincigrupegitim,ikincigrupegitim,ucuncugrupegitim,birincigrupirk,ikincigrupirk,ucuncugrupirk,birincigelir,ikincigelir,marital,own,employbir,employiki,censusbir,censusiki])
        if self.model1.predict(dizi)==1:
            self.label_37.setText('Vaccinated')
        else:
            self.label_37.setText('Not vaccinated')
        if self.model2.predict(dizi)==1:
            self.label_36.setText('Vaccinated')
        else:
            self.label_36.setText('Not vaccinated')
if __name__ == '__main__':
    app=QApplication(sys.argv)
    widget=Pencere()
    widget.show()
    app.exit(app.exec())