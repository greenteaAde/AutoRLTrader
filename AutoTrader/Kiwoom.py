import sys
from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
import time
import os
import sqlite3
import pandas as pd


class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()
        # 키움증권의 OpenAPI+를 사용하기 위한 COM 오브젝트 생성
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")
        #self.OnEventConnect.connect(self._event_connect)
        self.OnReceiveTrData.connect(self._receive_tr_data)

        self.deposit = None
        self.acc_info_1 = []
        self.acc_info_2 = {'종목명': [], '보유수량': [], '매입가': [], '현재가': [], '평가손익': [], '수익률(%)': []}

        self.ohlcv_data = {'일자': [], '시가': [], '고가': [], '저가': [], '현재가': [], '거래량': []}

    def reset_acc_info(self):
        self.deposit = None
        self.acc_info_1 = []
        self.acc_info_2 = {'종목명': [], '보유수량': [], '매입가': [], '현재가': [], '평가손익': [], '수익률(%)': []}

    def reset_ohlcv_data(self):
        self.ohlcv_data = {'일자': [], '시가': [], '고가': [], '저가': [], '현재가': [], '거래량': []}

    def comm_connect(self):
        # 로그인 윈도우 실행
        self.dynamicCall('CommConnect()')

    def _event_connect(self):
        pass

    def get_connect_state(self):
        # 접속 상태 반환
        return self.dynamicCall('GetConnectState()')

    def get_login_info(self, code):
        # 로그인 정보 반환
        info = self.dynamicCall('GetLoginInfo(QString)', [code])
        return info

    def get_server_gubun(self):
        # 실서버와 모의투자 서버를 구분하기 위한 메소드
        return self.dynamicCall("KOA_Functions(QString, QString)", "GetServerGubun", "")

    def get_code_list_by_market(self, market):
        # 장 구분 별 종목코드 리스트 반환
        code_list = self.dynamicCall("GetCodeListByMarket(QString)", [market])
        code_list = code_list.split(';')
        return code_list[:-1]

    def get_master_code_name(self, code):
        return self.dynamicCall('GetMasterCodeName(QString)', [code])

    def set_input_value(self, id, value):
        self.dynamicCall("SetInputValue(QString, QString)", id, value)

    def comm_rq_data(self, rqname, trcode, next, screen_no):
        print('comm_rq_data')
        self.dynamicCall("CommRqData(QString, QString, int, QString", rqname, trcode, next, screen_no)
        self.requestLoop = QEventLoop()
        self.requestLoop.exec_()

    def _comm_get_data(self, code, real_type, field_name, index, item_name):
        print('_comm_get_data')
        ret = self.dynamicCall("CommGetData(QString, QString, QString, int, QString", code,
                               real_type, field_name, index, item_name)
        return ret.strip()

    def _get_repeat_cnt(self, trcode, rqname):
        print('get_repeat_cnt')
        ret = self.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
        return ret

    def _receive_tr_data(self, screen_no, rqname, trcode, record_name, next):
        print('receive_tr_data')
        if next == '2':
            self.remained_data = True
        else:
            self.remained_data = False

        if rqname == "opt10081_req":
            self._opt10081_req(rqname, trcode)
        elif rqname == 'opw00001_req':
            self._opw00001_req(rqname, trcode)
        elif rqname == 'opw00018_req':
            self._opw00018_req(rqname, trcode)

        try:
            self.requestLoop.exit()
        except AttributeError:
            pass

    def _opw00001_req(self, rqname, trcode):
        self.deposit = self._comm_get_data(trcode, "", rqname, 0, "d+2추정예수금")

    def _opw00018_req(self, rqname, trcode):
        acc_name = ['총매입금액', '총평가금액', '총평가손익금액', '총수익률(%)', '추정예탁자산']
        for y in acc_name:
            self.acc_info_1.append(self._comm_get_data(trcode, "", rqname, 0, y))

        equity_cnt = self._get_repeat_cnt(trcode, rqname)
        equity_name = ['종목명', '보유수량', '매입가', '현재가', '평가손익', '수익률(%)']
        for x in range(equity_cnt):
            for y in equity_name:
                self.acc_info_2[y].append(self._comm_get_data(trcode, "", rqname, x, y))

    def _opt10081_req(self, rqname, trcode):
        data_cnt = self._get_repeat_cnt(trcode, rqname)
        data_name = ['일자', '시가', '고가', '저가', '현재가', '거래량']
        for x in range(data_cnt):
            for y in data_name:
                data = self._comm_get_data(trcode, "", rqname, x, y)
                self.ohlcv_data[y].append(data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    kiwoom = Kiwoom()
    kiwoom.comm_connect()

    # opt10081 TR 요청
    kiwoom.set_input_value("종목코드", "039490")
    kiwoom.set_input_value("기준일자", "20170224")
    kiwoom.set_input_value("수정주가구분", 1)
    kiwoom.comm_rq_data("opt10081_req", "opt10081", 0, "0101")

    while kiwoom.remained_data:
        time.sleep(0.2)
        kiwoom.set_input_value("종목코드", "039490")
        kiwoom.set_input_value("기준일자", "20170224")
        kiwoom.set_input_value("수정주가구분", 1)
        kiwoom.comm_rq_data("opt10081_req", "opt10081", 2, "0101")

    df = pd.DataFrame(kiwoom.ohlcv_data)
    df.index = df['일자']
    df = df.iloc[:, 1:]

    con = sqlite3.connect(os.getcwd()+'/stock.db')
    df.to_sql('039490', con, if_exists='replace')