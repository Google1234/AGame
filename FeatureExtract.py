'''
#user_feature from offline train file
import Config
import pandas
rats=pandas.read_csv(Config.path+"ccf_offline_stage1_train._sorted_by_User_id.csv",header=None)
feature_format=Config.user_feature_format()
file_format=Config.file_offline_train_line()
result=[]
result.append([0 for i in xrange(feature_format.len)])
line2=0
result[line2][feature_format.user_id] = rats.iloc[0][file_format.User_id]
print len(rats)/10000
for line in xrange(len(rats)):
    if rats.iloc[line][file_format.User_id]!=result[line2][feature_format.user_id]:
        if (result[line2][feature_format.not_coupon_consump]+
                result[line2][feature_format.coupon_consump])==0:
            result[line2][feature_format.coupon_consump_div_total_consump]=0.5
        else:
            result[line2][feature_format.coupon_consump_div_total_consump]=result[line2][feature_format.coupon_consump]*1.0\
                                                                       /(result[line2][feature_format.not_coupon_consump]+
                                                                         result[line2][feature_format.coupon_consump])
        if  result[line2][feature_format.total_coupon]==0:
            result[line2][feature_format.coupon_consump_div_total_coupon]=0.5
        else:
            result[line2][feature_format.coupon_consump_div_total_coupon] = result[line2][
                                                                             feature_format.coupon_consump] * 1.0 \
                                                                         / (result[line2][feature_format.total_coupon])
        result.append([0 for i in xrange(feature_format.len)])
        line2+=1
        result[line2][feature_format.user_id] = rats.iloc[line][file_format.User_id]
    if rats.iloc[line][file_format.Date]!="null":
        if rats.iloc[line][file_format.Date_received]!="null":
            result[line2][feature_format.coupon_consump]+=1
        else:
            result[line2][feature_format.not_coupon_consump]+=1
    if rats.iloc[line][file_format.Date_received]!="null":
        result[line2][feature_format.total_coupon]+=1
    if line%10000==0:
        print line/10000

if (result[line2][feature_format.not_coupon_consump] +
        result[line2][feature_format.coupon_consump]) == 0:
    result[line2][feature_format.coupon_consump_div_total_consump] = 0.5
else:
    result[line2][feature_format.coupon_consump_div_total_consump] = result[line2][feature_format.coupon_consump] * 1.0 \
                                                                     / (
                                                                     result[line2][feature_format.not_coupon_consump] +
                                                                     result[line2][feature_format.coupon_consump])
if result[line2][feature_format.total_coupon] == 0:
    result[line2][feature_format.coupon_consump_div_total_coupon] = 0.5
else:
    result[line2][feature_format.coupon_consump_div_total_coupon] = result[line2][
                                                                        feature_format.coupon_consump] * 1.0 \
                                                                    / (result[line2][feature_format.total_coupon])
pandas.DataFrame(result).to_csv(Config.path+Config.feature_path+"User_from_offline_train.csv",index=None,header=None)
'''

'''
#merchant feature from offline train file
import Config
import pandas
rats=pandas.read_csv(Config.path+"ccf_offline_stage1_train._sorted_by_Merchant_id.csv",header=None)
#rats=pandas.read_csv("test.csv",header=None)
feature_format=Config.merchant_feature_format()
file_format=Config.file_offline_train_line()
result=[]
result.append([0 for i in xrange(feature_format.len)])
line2=0
result[line2][feature_format.merchant_id] = rats.iloc[0][file_format.Merchant_id]
print len(rats)/10000
for line in xrange(len(rats)):
    if rats.iloc[line][file_format.Merchant_id]!=result[line2][feature_format.merchant_id]:
        if (result[line2][feature_format.not_coupon_consump]+
                result[line2][feature_format.coupon_consump])==0:
            result[line2][feature_format.coupon_consump_div_total_consump]=0.5
        else:
            result[line2][feature_format.coupon_consump_div_total_consump]=result[line2][feature_format.coupon_consump]*1.0\
                                                                       /(result[line2][feature_format.not_coupon_consump]+
                                                                         result[line2][feature_format.coupon_consump])
        if  result[line2][feature_format.total_coupon]==0:
            result[line2][feature_format.coupon_consump_div_total_coupon]=0.5
        else:
            result[line2][feature_format.coupon_consump_div_total_coupon] = result[line2][
                                                                             feature_format.coupon_consump] * 1.0 \
                                                                         / (result[line2][feature_format.total_coupon])
        result.append([0 for i in xrange(feature_format.len)])
        line2+=1
        result[line2][feature_format.merchant_id] = rats.iloc[line][file_format.Merchant_id]
    if rats.iloc[line][file_format.Date]!="null":
        if rats.iloc[line][file_format.Date_received]!="null":
            result[line2][feature_format.coupon_consump]+=1
        else:
            result[line2][feature_format.not_coupon_consump]+=1
    if rats.iloc[line][file_format.Date_received]!="null":
        result[line2][feature_format.total_coupon]+=1
    if line%10000==0:
        print line/10000

if (result[line2][feature_format.not_coupon_consump] +
        result[line2][feature_format.coupon_consump]) == 0:
    result[line2][feature_format.coupon_consump_div_total_consump] = 0.5
else:
    result[line2][feature_format.coupon_consump_div_total_consump] = result[line2][feature_format.coupon_consump] * 1.0 \
                                                                     / (
                                                                     result[line2][feature_format.not_coupon_consump] +
                                                                     result[line2][feature_format.coupon_consump])
if result[line2][feature_format.total_coupon] == 0:
    result[line2][feature_format.coupon_consump_div_total_coupon] = 0.5
else:
    result[line2][feature_format.coupon_consump_div_total_coupon] = result[line2][
                                                                        feature_format.coupon_consump] * 1.0 \
                                                                    / (result[line2][feature_format.total_coupon])
pandas.DataFrame(result).to_csv(Config.path+Config.feature_path+"Merchant_from_offline_train.csv",index=None,header=None)
'''

import Config
import pandas

#rats=pandas.read_csv(Config.path+"ccf_offline_stage1_train._sorted_by_Coupon_id.csv",header=None)
rats=pandas.read_csv("test.csv",header=None)
feature_format=Config.coupon_feature_format()
file_format=Config.file_offline_train_line()
result=[]
result.append([0 for i in xrange(feature_format.len)])
line2=0
result[line2][feature_format.coupon_id] = rats.iloc[0][file_format.Coupon_id]
print len(rats)/10000
for line in xrange(len(rats)):
    if rats.iloc[line][file_format.Coupon_id]=="null":
        break
    if rats.iloc[line][file_format.Coupon_id]!=result[line2][feature_format.coupon_id]:
        if (result[line2][feature_format.not_coupon_consump]+
                result[line2][feature_format.coupon_consump])==0:
            result[line2][feature_format.coupon_consump_div_total_consump]=0.5
        else:
            result[line2][feature_format.coupon_consump_div_total_consump]=result[line2][feature_format.coupon_consump]*1.0\
                                                                       /(result[line2][feature_format.not_coupon_consump]+
                                                                         result[line2][feature_format.coupon_consump])
        if  result[line2][feature_format.total_coupon]==0:
            result[line2][feature_format.coupon_consump_div_total_coupon]=0.5
        else:
            result[line2][feature_format.coupon_consump_div_total_coupon] = result[line2][
                                                                             feature_format.coupon_consump] * 1.0 \
                                                                         / (result[line2][feature_format.total_coupon])
        result.append([0 for i in xrange(feature_format.len)])
        line2+=1
        result[line2][feature_format.coupon_id] = rats.iloc[line][file_format.Coupon_id]
        if rats.iloc[line][file_format.Discount_rate]
    if rats.iloc[line][file_format.Date]!="null":
        if rats.iloc[line][file_format.Date_received]!="null":
            result[line2][feature_format.coupon_consump]+=1
        else:
            result[line2][feature_format.not_coupon_consump]+=1
    if rats.iloc[line][file_format.Date_received]!="null":
        result[line2][feature_format.total_coupon]+=1
    if line%10000==0:
        print line/10000

if (result[line2][feature_format.not_coupon_consump] +
        result[line2][feature_format.coupon_consump]) == 0:
    result[line2][feature_format.coupon_consump_div_total_consump] = 0.5
else:
    result[line2][feature_format.coupon_consump_div_total_consump] = result[line2][feature_format.coupon_consump] * 1.0 \
                                                                     / (
                                                                     result[line2][feature_format.not_coupon_consump] +
                                                                     result[line2][feature_format.coupon_consump])
if result[line2][feature_format.total_coupon] == 0:
    result[line2][feature_format.coupon_consump_div_total_coupon] = 0.5
else:
    result[line2][feature_format.coupon_consump_div_total_coupon] = result[line2][
                                                                        feature_format.coupon_consump] * 1.0 \
                                                                    / (result[line2][feature_format.total_coupon])
pandas.DataFrame(result).to_csv(Config.path+Config.feature_path+"Merchant_from_offline_train.csv",index=None,header=None)
