import pandas as pd
import os
import warnings
import pickle


warnings.filterwarnings('ignore')

class etc_handler(object):
    def __init__(self,path_root=''):
        self.path_root = path_root
        self.filename = 'tasharep.csv'
        self.picklename = 'tasharep.pickle'
        self.data_file_path = os.path.join(path_root,self.filename)
        self.pickle_file_path = os.path.join(path_root,self.picklename)
        if os.path.isfile(self.pickle_file_path):
            self.etc_data = pickle.load(open(self.pickle_file_path, 'rb'))
        else:
            self.etc_data = self.data_convert(pd.read_csv(self.data_file_path,encoding='hkscs', skipinitialspace=True))

    def data_convert(self, input_data):
        etc_dict = dict()
        input_data.columns = ['code', 'date', 'chinese_name', 'open', 'high', 'low', 'close', 'ammount']
        all_code_list = pd.unique(input_data['code'])
        for code in all_code_list:
            tmp_dict = dict()
            single_code = input_data.loc[input_data['code'] == code]
            for index, row in single_code.iterrows():
                date = row['date']
                price = row[3:].tolist()
                price[-1] = int(price[-1].strip().replace(',',''))
                tmp_dict[date]=price
            etc_dict[code] = tmp_dict
        pickle.dump(etc_dict,open(self.pickle_file_path, 'wb'))
        return etc_dict
            
    def get_price(self, code_number):
        code_data = self.etc_data[code_number]
        if self.check_sort(list(code_data.keys())):
            price_data = list(code_data.values())
        else:
            price_data = self.get_sorted_price(list(code_data.keys()),code_data)
        return price_data
    
    def check_sort(self, keys):
        return keys == sorted(keys)

    def get_sorted_price(self,keys,code_data):
        tmp_price_data = list()
        keys = sorted(keys)
        for key in keys:
            tmp_price_data.append(list(code_data[key]))
        return tmp_price_data

if __name__ == '__main__':
    etc_data = etc_handler('I:\ETC\TBrain_Round2_DataSet_20180331')
    print (len(etc_data.get_price(1101)))
    

