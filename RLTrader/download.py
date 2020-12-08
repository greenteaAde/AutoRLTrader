# %%writefile __download__.py

" Excel file if firm lists with firm symbol named KOSPI.xls, KOSDAQ.xls, KONEX.xls shold be in the same path"

import pandas as pd
import requests
import numpy as np
import time
import bs4


def financial_statement(symbol):
    data_url = 'http://comp.fnguide.com/SVO2/ASP/SVD_Finance.asp?pGB=1&gisymbol=' + symbol + '&cID=&MenuYn=Y&ReportGB=D&NewMenuID=103&stkGb=701'
    data_page = requests.get(data_url)
    data_table = pd.read_html(data_page.text)

    df1 = data_table[0].set_index('IFRS(연결)' or 'GAPP' or 'GAAP(개별)').iloc[:, :4].loc[['매출액', '영업이익', '당기순이익']]
    df2 = data_table[2].set_index('IFRS(연결)' or 'GAPP' or 'GAAP(개별)').loc[['자산', '부채', '자본']]
    df3 = data_table[4].set_index('IFRS(연결)' or 'GAPP' or 'GAAP(개별)').loc[['영업활동으로인한현금흐름']]

    data_df = pd.concat([df1, df2, df3])

    return data_df


def financial_ratio(symbol):
    data_url = 'http://comp.fnguide.com/SVO2/ASP/SVD_FinanceRatio.asp?pGB=1&gisymbol=' + symbol + '&cID=&MenuYn=Y&ReportGB=D&NewMenuID=103&stkGb=701'
    data_page = requests.get(data_url)
    data_table = pd.read_html(data_page.text)

    df = data_table[0].set_index('IFRS(연결)' or 'GAPP' or 'GAAP(개별)')
    data_df = df.loc[['유동비율계산에 참여한 계정 펼치기', '부채비율계산에 참여한 계정 펼치기', '영업이익률계산에 참여한 계정 펼치기',
                      '영업이익률계산에 참여한 계정 펼치기', 'ROIC계산에 참여한 계정 펼치기']]
    data_df.index = ['유동비율', '부채비율', '영업이익률', 'ROA', 'ROIC']

    return data_df


def invest_indicator(symbol):
    data_url = 'http://comp.fnguide.com/SVO2/ASP/SVD_Invest.asp?pGB=1&gisymbol=' + symbol + '&cID=&MenuYn=Y&ReportGB=D&NewMenuID=103&stkGb=701'
    data_page = requests.get(data_url)
    data_table = pd.read_html(data_page.text)

    df = data_table[1].set_index('IFRS 연결' or 'GAPP' or 'GAAP(개별)')
    data_df = df.loc[['PER계산에 참여한 계정 펼치기', 'PCR계산에 참여한 계정 펼치기', 'PSR계산에 참여한 계정 펼치기',
                      'PBR계산에 참여한 계정 펼치기', '총현금흐름']]
    data_df.index = ['PER', 'PCR', 'PSR', 'PBR', '총현금흐름']

    return data_df


def get_symbol(*firm_name_files, data='each'):
    symbol_df = pd.DataFrame([])
    for num, file_name in enumerate(firm_name_files):
        symbol_list = pd.read_excel(file_name).loc[:, ['종목코드']]
        symbol_list['종목코드'] = symbol_list['종목코드'].apply(
            lambda x: 'A' + '0' * (6 - len(str(x))) + str(x) if x != np.nan else x)
        symbol_list.columns = [str(file_name)]
        if data == 'each':
            if num == 0:
                symbol_df = symbol_list
            else:
                symbol_df = symbol_df.merge(symbol_list, how='outer', left_index=True, right_index=True)
        elif data == 'all':
            if num == 0:
                symbol_df = symbol_list
            else:
                symbol_df = symbol_df.concat(symbol_list, axis=0)

    return symbol_df


def change_df(symbol, dataframe):
    for num, col in enumerate(dataframe.columns):
        tmp = pd.DataFrame({symbol: dataframe[col]})
        tmp = tmp.T
        tmp.columns = [[col] * len(dataframe), tmp.columns]
        if num == 0:
            multicolumn_df = tmp
        else:
            multicolumn_df = pd.merge(multicolumn_df, tmp, how='outer', left_index=True, right_index=True)

    return multicolumn_df


def download(firm_list, data='fs'):
    """
    data
    'fs': financial statement
    'fr': financial ratio
    'ii': invest_indicator
    """
    for num, symbol in enumerate(firm_list):
        try:
            print(num, end='  ')
            try:
                if data == 'fs':
                    raw_data = financial_statement(symbol)
                elif data == 'fr':
                    raw_data = financial_ratio(symbol)
                elif data == 'ii':
                    raw_data = invest_indicator(symbol)

            except ValueError:
                print("'data' should be 'fs(financial_statement)' or 'fr(financial_ratio)' or 'ii(invest_indicator)'")

            except requests.exceptions.Timeout:
                time.sleep(30)
                if data == 'fs':
                    raw_data = financial_statement(symbol)
                elif data == 'fr':
                    raw_data = financial_ratio(symbol)
                elif data == 'ii':
                    raw_data = invest_indicator(symbol)

            df_changed = change_df(symbol, raw_data)
            if num == 0:
                final_df = df_changed
            else:
                final_df = pd.concat([final_df, df_changed], axis=1)

        except ValueError:
            continue

        except KeyError:
            continue

        except TypeError:
            break

    return final_df


def get_price(symbol, count):
    price_url = 'https://fchart.stock.naver.com/sise.nhn?symbol=' + symbol + '&timeframe=' + 'day' + '&count=' + count + '&requestType=0'
    price_data = requests.get(price_url)
    item_list = bs4.BeautifulSoup(price_data.text, 'lxml').find_all('item')

    date_list = []
    price_list = []

    for item in item_list:
        tmp_data = item['data'].split('|')
        date_list.append(tmp_data[0])
        price_list.append(tmp_data[4])

    price_df = pd.DataFrame({symbol: price_list}, index=date_list)
    price_df.index = pd.to_datetime(price_df.index)

    return price_df
