import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore
from PyQt5.QtCore import *
from Kiwoom import *
from qt_utils import *
from PyQt5 import uic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import sqlite3

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
plt.style.use('seaborn-whitegrid')

form_class = uic.loadUiType("Window.ui")[0]


class TradingWindow(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.kiwoom = Kiwoom()

        self.setupUi(self)
        self.setWindowTitle('강화학습 자동매매 프로그램')
        self.setWindowIcon(QIcon('icon.png'))

        # Main Window Status Bar (메인 윈도우 상태바)
        self.statusBar().showMessage('Ready')

        # Tab Name 변경
        self.tabWidget.setTabText(self.tabWidget1.indexOf(self.tab1), '계좌정보')
        self.tabWidget.setTabText(self.tabWidget1.indexOf(self.tab2), '수동매매')
        self.tabWidget.setTabText(self.tabWidget1.indexOf(self.tab3), '강화학습')

        self.tabWidget2.setTabText(self.tabWidget2.indexOf(self.tab2_1), '차트')
        self.tabWidget2.setTabText(self.tabWidget1.indexOf(self.tab2_2), '테이블')
        self.tabWidget2.setTabText(self.tabWidget1.indexOf(self.tab2_3), 'DB저장')

        # Main Window Menu Bar-Login (메인 윈도우 메뉴바 로그인)
        login_action = QAction(QIcon('exit.png'), 'Login', self)
        login_action.setShortcut('Ctrl+L')
        login_action.setStatusTip('Login')
        login_action.triggered.connect(self.login)

        # Main Window Menu Bar-Connect (메인 윈도우 메뉴바 연결상태 확인)
        status_action = QAction(QIcon('exit.png'), 'Connect Status', self)
        status_action.setShortcut('Ctrl+S')
        status_action.setStatusTip('Show Connect Status')
        status_action.triggered.connect(self.connect_state)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        file_menu = menubar.addMenu('&File')  # Ctrl+F
        file_menu.addAction(login_action)
        file_menu.addAction(status_action)

        self.kiwoom.OnEventConnect.connect(self.get_accounts)  # 키움 서버 접속 관련 이벤트가 발생할 경우 event_connect 함수 호출

        # 시장선택
        self.radio_Exchange.clicked.connect(self.load_Exchange)
        self.radio_KOTC.clicked.connect(self.load_KOTC)
        self.radio_KOSDAQ.clicked.connect(self.load_KOSDAQ)

        # 기준일자 설정
        self.dateEdit.setCalendarPopup(True)

        # 조회버튼
        self.stockSearch.clicked.connect(self.lookup)

        # test lineEdit
        self.textEdit = QPlainTextEdit(self.tab1_1)

        # 계좌를 조회하는 PushButton
        self.pushAccount.clicked.connect(self.get_account_info)

    def login(self):
        # 로그인 버튼을 누르면 로그인을 합니다.
        self.kiwoom.comm_connect()

    def get_accounts(self, err_code):
        if err_code == 0:
            # 계좌 선택 콤보박스를 채웁니다.
            account_num = int(self.kiwoom.get_login_info("ACCOUNT_CNT"))
            account = self.kiwoom.get_login_info("ACCNO")
            account_list = account.split(';')[0:account_num]
            self.comboAccount.addItems(account_list)
        else:
            print('failed')

    def get_account_info(self):
        self.kiwoom.reset_acc_info()

        account_num = self.comboAccount.currentText()
        self.kiwoom.set_input_value("계좌번호", account_num)
        self.kiwoom.comm_rq_data("opw00018_req", "opw00018", 0, '2000')

        while self.kiwoom.remained_data:
            time.sleep(0.2)
            self.kiwoom.set_input_value("계좌번호", account_num)
            self.kiwoom.comm_rq_data("opw00018_req", "opw00018", 0, '2000')

        self.kiwoom.set_input_value("계좌번호", account_num)
        self.kiwoom.comm_rq_data("opw00001_req", "opw00001", 0, '2000')

        self.tableAccount1.setRowCount(1)
        item = QTableWidgetItem(utils.change_format(self.kiwoom.deposit))
        item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
        self.tableAccount1.setItem(0, 0, item)

        for i in range(1, 6):
            item = QTableWidgetItem(utils.change_format(self.kiwoom.acc_info_1[i-1]))
            item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
            self.tableAccount1.setItem(0, i, item)
        self.tableAccount1.resizeRowsToContents()

        item_count = len(self.kiwoom.acc_info_2['종목명'])
        self.tableAccount2.setRowCount(item_count)
        for i in range(item_count):
            for j, v in enumerate(self.kiwoom.acc_info_2.values()):
                item = QTableWidgetItem(utils.change_format(v[i]))
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self.tableAccount2.setItem(i, j, item)

        self.tableAccount2.resizeRowsToContents()

    def connect_state(self):
        # 현재 Kiwoom OpenAPI+에 연결되었는지를 확인합니다.
        if self.kiwoom.get_connect_state() == 0:
            self.statusBar().showMessage("Not Connected")
        else:
            self.statusBar().showMessage("Connected")

    """주식 조회 전 설정 관련 메소드"""
    def load_Exchange(self):
        self.comboBox.clear()
        self.load_market_stocks('0')

    def load_KOTC(self):
        self.comboBox.clear()
        self.load_market_stocks('30')

    def load_KOSDAQ(self):
        self.comboBox.clear()
        self.load_market_stocks('10')

    def load_market_stocks(self, num_str):
        # 시장선택을 하면 종목코드를 콤보박스로 가져옵니다.
        code_list = self.kiwoom.get_code_list_by_market(num_str)
        code_name_list = []
        for code in code_list:
            name = self.kiwoom.get_master_code_name(code)
            code_name_list.append(code + ' : ' + name)
        self.comboBox.addItems(code_name_list)

    def lookup(self):
        self.kiwoom.reset_ohlcv_data()

        code = self.comboBox.currentText()[:6]
        date = self.dateEdit.date().toString('yyyyMMdd')

        self.kiwoom.set_input_value("종목코드", code)
        self.kiwoom.set_input_value("기준일자", date)
        self.kiwoom.set_input_value("수정주가구분", 1)
        self.kiwoom.comm_rq_data("opt10081_req", "opt10081", 0, "0101")

        try:
            while kiwoom.remained_data:
                time.sleep(0.2)
                self.kiwoom.set_input_value("종목코드", code)
                self.kiwoom.set_input_value("기준일자", date)
                self.kiwoom.set_input_value("수정주가구분", 1)
                self.kiwoom.comm_rq_data("opt10081_req", "opt10081", 2, "0101")
        except Exception as exception:
            print('example:', type(exception).__name__)

        _df = pd.DataFrame(self.kiwoom.ohlcv_data)
        _df['일자'] = _df['일자'].map(lambda x: int(x))
        _df.index = _df['일자']
        _df = _df.iloc[:, 1:]
        _df = _df.sort_index(ascending=True)
        _df.index = _df.index.map(lambda x: str(x))

        _con = sqlite3.connect(f'{os.getcwd()}/{code}.db')
        _df.to_sql(code, _con, if_exists='replace')






    def init_ui(self):




        self.dateEdit = QDateEdit(self)

        self.dateEdit.setGeometry(QtCore.QRect(70, 240, 531, 21))
        self.dateEdit.setObjectName("기준일자")

        self.tableView = QTableView(self)
        self.tableView.setGeometry(QtCore.QRect(70, 280, 400, 200))

        # self.lookupButton = QPushButton(self)
        # self.lookupButton.setObjectName('조회')
        #
        # self.lookupButton.setGeometry(QtCore.QRect(70, 320, 21, 21))

        self.show()






if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TradingWindow()
    ex.show()
    sys.exit(app.exec_())