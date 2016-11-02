import Config
import pandas
from multiprocessing import Pool
import FeatureExtract
#import tensorflow as tf

#def split_file(input_dir,input_file_name,output_dir,output_file_name,parts):
'''
dic={}
dic["input_dir"]="Data/tmp/"
dic["input_file_name"]="test.csv"
dic["parts"]=4
dic["output_dir"]="Data/tmp/"
dic["output_file_name"]="test.csv"
'''
def split_file(dic):
    input_dir=dic["input_dir"]
    input_file_name=dic["input_file_name"]
    parts=dic["parts"]
    output_dir=dic["output_dir"]
    output_file_name=dic["output_file_name"]

    rats=pandas.read_csv(input_dir+input_file_name,header=None)
    remain=len(rats)
    last_line=0
    line=(remain+(parts-1))/parts
    remain-=((remain+(parts-1))/parts)
    parts-=1
    id=1
    names=[]
    #if tf.gfile.Exists(output_dir):
        #tf.gfile.DeleteRecursively(output_dir)
    #tf.gfile.MakeDirs(output_dir)
    while parts>0:
        rats.iloc[last_line:line].to_csv(output_dir+str(id)+output_file_name,index=None,header=None)
        names.append(str(id)+output_file_name)
        last_line=line
        line+=((remain+(parts-1))/parts)
        remain -= ((remain + (parts - 1)) / parts)
        parts-=1
        id+=1
    rats.iloc[last_line:].to_csv(output_dir+str(id) + output_file_name, index=None, header=None)
    names.append(str(id)+output_file_name)
    return output_dir,names

#def make(input_dir,input_file_name,output_dir,output_dir_name,file_format=Config.file_offline_train_line()):
'''
dic={}
dic["input_dir"]=_dir
dic["input_file_name"] = names[i]
dic["file_format"]=Config.file_offline_train_line()
dic["input_dir"] = _dir
dic["output_file_name"] = output_file_name
'''
def make_samples(dic):
    input_dir=dic["input_dir"]
    input_file_name=dic["input_file_name"]
    file_format=dic["file_format"]
    output_dir=dic["output_dir"]
    output_file_name=dic["output_file_name"]

    rats=pandas.read_csv(input_dir+input_file_name,header=None)
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
    #if tf.gfile.Exists(output_dir):
        #tf.gfile.DeleteRecursively(output_dir)
    #tf.gfile.MakeDirs(output_dir)
    pandas.DataFrame(postive_sample).to_csv(output_dir+"postive_sample_"+output_file_name,index=None,header=None)
    pandas.DataFrame(negtive_sample).to_csv(output_dir+"negtive_sample_"+output_file_name,index=None,header=None)
    pandas.DataFrame(consump_not_use_coupon_sample).to_csv(output_dir+"consump_not_use_coupon_sample_"+output_file_name,index=None,header=None)
    return output_dir,["postive_sample_"+output_file_name,"negtive_sample_"+output_file_name,"consump_not_use_coupon_sample_"+output_file_name]

#def merge_files(input_dir,input_files_names,output_dir,output_name):
'''
dic={}
dic["input_dir"]=_dir
dic["input_files_names"]=names
dic["output_dir"]="Data/tmp/"
dic["output_name"]="merge.csv"
'''
def merge_files(dic):
    input_dir=dic["input_dir"]
    input_files_names=dic["input_files_names"]
    output_dir=dic["output_dir"]
    output_name=dic["output_name"]

    frames=[]
    for file in input_files_names:
        frames.append(pandas.read_csv(input_dir+file,header=None))
    #if tf.gfile.Exists(output_dir):
        #tf.gfile.DeleteRecursively(output_dir)
    #tf.gfile.MakeDirs(output_dir)
    pandas.concat(frames).to_csv(output_dir+output_name,index=None,header=None)
    return output_dir,output_name

def parallel_make(process_numbers):
    #split files
    dic = {}
    dic["input_dir"] = "Data/tmp/"
    dic["input_file_name"] = "test.csv"
    dic["parts"] = 4
    dic["output_dir"] = "Data/tmp/"
    dic["output_file_name"] = "test.csv"
    output_dir,names=split_file(dic)
    #make samples
    dicts=[{} for i in range(len(names))]
    for i in range(len(names)):
        dicts[i]["input_dir"]=output_dir
        dicts[i]["input_file_name"] = names[i]
        dicts[i]["file_format"]=Config.file_offline_train_line()
        dicts[i]["output_dir"] =output_dir
        dicts[i]["output_file_name"] = ".csv"
    pool = Pool(process_numbers)
    print pool.map(make_samples,dicts)
    pool.close()
    pool.join()
    #merge sample files to one file
    pool=Pool()