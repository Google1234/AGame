import Config
import pandas
import FeatureExtract
import tensorflow as tf
def split_file(_dir,file_name,parts,file_format=Config.file_offline_train_line):
    rats=pandas.read_csv(_dir+file_name,header=None)
    remain=len(rats)
    last_line=0
    line=(remain+(parts-1))/parts
    remain-=((remain+(parts-1))/parts)
    parts-=1
    id=1
    names=[]
    if tf.gfile.Exists(_dir+"tmp/"):
        tf.gfile.DeleteRecursively(_dir+"tmp/")
    tf.gfile.MakeDirs(_dir+"tmp/")
    while parts>0:
        rats.iloc[last_line:line].to_csv(_dir+"tmp/"+str(id)+file_name,index=None,header=None)
        names.append(str(id)+file_name)
        last_line=line
        line+=((remain+(parts-1))/parts)
        remain -= ((remain + (parts - 1)) / parts)
        parts-=1
        id+=1
    rats.iloc[last_line:].to_csv(_dir+"tmp/" +str(id) + file_name, index=None, header=None)
    names.append(str(id)+file_name)
    return _dir+"tmp/",names
#def make(_dir,file,file_format=Config.file_offline_train_line()):
def make(dic):
    dic={}
    _dir=dic["_dir"]
    file=dic["file"]
    if "file_format" in dic.keys():
        file_format=dic["file_format"]
    else:
        file_format=Config.file_offline_train_line()

    rats=pandas.read_csv(_dir+file,header=None)
    postive_sample=[]
    negtive_sample=[]
    consump_not_use_coupon_sample=[]
    user_feature=FeatureExtract.user_feature(Config.path+Config.feature_path+"User_from_offline_train.csv",False)
    merchant_feature=FeatureExtract.merchant_feature(Config.path+Config.feature_path+"Merchant_from_offline_train.csv",False)
    coupon_feature=FeatureExtract.coupon_feature(Config.path+Config.feature_path+"Coupon_from_offline_train.csv",False)
    distance=0
    print "total :",len(rats)/10000,"*10000"
    for line in xrange(len(rats)):
        if rats.iloc[line][file_format.Date_received]!="null" and rats.iloc[line][file_format.Date]!="null":
            if rats.iloc[line][file_format.Distance]=="null":
                distance=Config.avr_distance
            else:
                distance=int(rats.iloc[line][file_format.Distance])
            postive_sample.append(
                user_feature.get(rats.iloc[line][file_format.User_id])[1:]+
                merchant_feature.get(rats.iloc[line][file_format.Merchant_id])[1:] +
                coupon_feature.get(rats.iloc[line][file_format.Coupon_id])[1:] +
                [distance]
            )
        if rats.iloc[line][file_format.Date_received]=='null' and rats.iloc[line][file_format.Date]!="null":
            if rats.iloc[line][file_format.Distance]=="null":
                distance=Config.avr_distance
            else:
                distance=int(rats.iloc[line][file_format.Distance])
            consump_not_use_coupon_sample.append(
                user_feature.get(rats.iloc[line][file_format.User_id])[1:]+
                merchant_feature.get(rats.iloc[line][file_format.Merchant_id])[1:] +
                coupon_feature.get(rats.iloc[line][file_format.Coupon_id])[1:] +
                [distance]
            )
        if rats.iloc[line][file_format.Date_received]!="null" and rats.iloc[line][file_format.Date]=="null":
            if rats.iloc[line][file_format.Distance]=="null":
                distance=Config.avr_distance
            else:
                distance=int(rats.iloc[line][file_format.Distance])
            negtive_sample.append(
                user_feature.get(rats.iloc[line][file_format.User_id])[1:]+
                merchant_feature.get(rats.iloc[line][file_format.Merchant_id])[1:] +
                coupon_feature.get(rats.iloc[line][file_format.Coupon_id])[1:]+
                [distance]
            )
            if line%10000==0:
                print "Finish ",line/10000,"*10000"
    pandas.DataFrame(postive_sample).to_csv(_dir+"postive_sample_"+file,index=None,header=None)
    pandas.DataFrame(negtive_sample).to_csv(_dir+"negtive_sample_"+file,index=None,header=None)
    pandas.DataFrame(consump_not_use_coupon_sample).to_csv(_dir+"consump_not_use_coupon_sample_"+file,index=None,header=None)
    return _dir,["postive_sample_"+file,"negtive_sample_"+file,"consump_not_use_coupon_sample_"+file]

def parallel_make(process_numbers):
    _dir,names=split_file("","test.csv",process_numbers)
    dicts=[{} for i in range(len(names))]
    for i in range():
        dicts[]
    from multiprocessing import Pool
    pool = Pool(process_numbers)
    pool.map(make,names)
    pool.close()
    pool.join()