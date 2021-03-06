#sort by User_id or else
import pandas
import Config
def sort_all():
    file="ccf_offline_stage1_train.csv"
    file_format=Config.file_offline_train_line()
    rats = pandas.read_csv(Config.path+file,header=None)
    for key in file_format.dic :
        rats.sort_values(by=file_format.dic[key],inplace=True)#user_id
        rats.to_csv(Config.path+file[:-3]+"_sorted_by_"+key+".csv",index=None,header=None)
    del rats

    file="ccf_online_stage1_train.csv"
    file_format=Config.file_online_train_line()
    rats = pandas.read_csv(Config.path+file,header=None)
    for key in file_format.dic :
        rats.sort_values(by=file_format.dic[key],inplace=True)#user_id
        rats.to_csv(Config.path+file[:-3]+"_sorted_by_"+key+".csv",index=None,header=None)
    del rats

    file="ccf_offline_stage1_test_revised.csv"
    file_format=Config.file_offline_test_line()
    rats = pandas.read_csv(Config.path+file,header=None)
    for key in file_format.dic :
        rats.sort_values(by=file_format.dic[key],inplace=True)#user_id
        rats.to_csv(Config.path+file[:-3]+"_sorted_by_"+key+".csv",index=None,header=None)
    del rats
def cal_avr_distance(file,file_format):
    rats=pandas.read_csv(file,header=None)
    user_id=-1
    merchant = set()
    add=0.0
    count=0
    print len(rats)/10000
    for line in xrange(len(rats)):
        if rats.iloc[line][file_format.User_id]!=user_id:
            merchant.clear()
            user_id=rats.iloc[line][file_format.User_id]
            if rats.iloc[line][file_format.Distance]!="null":
                merchant.add(rats.iloc[line][file_format.Merchant_id])
                add+=int(rats.iloc[line][file_format.Distance])
                count+=1
        else:
            if rats.iloc[line][file_format.Merchant_id] not in merchant and rats.iloc[line][file_format.Distance]!="null":
                merchant.add(rats.iloc[line][file_format.Merchant_id])
                add+=int(rats.iloc[line][file_format.Distance])
                count+=1
        if line%10000==0:
            print line/10000
            print ("fffff:",add,count)
    return add/count