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
