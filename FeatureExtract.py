import Config
import pandas
def bulid_user_feature():
    #user_feature from offline train file
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

def bulid_merchant_feature():
    #merchant feature from offline train file
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

def bulid_coupon_feature():
    import re
    FULL_CUT=1
    DISCOUNT=0

    rats=pandas.read_csv(Config.path+"ccf_offline_stage1_train._sorted_by_Coupon_id.csv",header=None)
    #rats=pandas.read_csv("test.csv",header=None)
    feature_format=Config.coupon_feature_format()
    file_format=Config.file_offline_train_line()
    result=[]
    result.append([0 for i in range(feature_format.len)])
    line2=0

    result[line2][feature_format.coupon_id] = rats.iloc[0][file_format.Coupon_id]
    if re.match(pattern=".*:", string=rats.iloc[0][file_format.Discount_rate]):
        result[line2][feature_format.fullCut_or_discount] = FULL_CUT
        list = rats.iloc[0][file_format.Discount_rate].split(":")
        result[line2][feature_format.discount] = (int(list[1]) * 1.0 / int(list[0]))
        result[line2][feature_format.full] = int(list[0])
    else:
        result[line2][feature_format.fullCut_or_discount] = DISCOUNT
        result[line2][feature_format.discount] = float(rats.iloc[0][file_format.Discount_rate])
        result[line2][feature_format.full] = 0

    print len(rats)/10000
    for line in xrange(len(rats)):
        if rats.iloc[line][file_format.Coupon_id]=="null":
            break
        if rats.iloc[line][file_format.Coupon_id]!=result[line2][feature_format.coupon_id]:
            if (result[line2][feature_format.used_coupon]+
                    result[line2][feature_format.not_used_coupon])==0:
                result[line2][feature_format.use_ratio]=0.5
            else:
                result[line2][feature_format.use_ratio]=result[line2][feature_format.used_coupon]*1.0\
                                                                           /(result[line2][feature_format.used_coupon]+
                                                                             result[line2][feature_format.not_used_coupon])
            result.append([0 for i in range(feature_format.len)])
            line2+=1
            result[line2][feature_format.coupon_id] = rats.iloc[line][file_format.Coupon_id]
            if re.match(pattern=".*:",string=rats.iloc[line][file_format.Discount_rate]):
                result[line2][feature_format.fullCut_or_discount]=FULL_CUT
                list = rats.iloc[line][file_format.Discount_rate].split(":")
                result[line2][feature_format.discount]=(int(list[1])*1.0/int(list[0]))
                result[line2][feature_format.full]=int(list[0])
            else:
                result[line2][feature_format.fullCut_or_discount] = DISCOUNT
                result[line2][feature_format.discount] =float(rats.iloc[line][file_format.Discount_rate])
                result[line2][feature_format.full] = 0

        if rats.iloc[line][file_format.Date]!="null":
            result[line2][feature_format.used_coupon]+=1
        else:
            result[line2][feature_format.not_used_coupon]+=1
        if  line %10000==0:
            print line/10000

    if (result[line2][feature_format.used_coupon] +
            result[line2][feature_format.not_used_coupon]) == 0:
        result[line2][feature_format.use_ratio] = 0.5
    else:
        result[line2][feature_format.use_ratio] = result[line2][feature_format.used_coupon] * 1.0 \
                                                  / (result[line2][feature_format.used_coupon] +
                                                     result[line2][feature_format.not_used_coupon])

    pandas.DataFrame(result,index=None).to_csv(Config.path+Config.feature_path+"Coupon_from_offline_train111.csv",index=None,header=None)
    sort_before=pandas.read_csv(Config.path+Config.feature_path+"Coupon_from_offline_train111.csv",header=None)
    sort_before.sort_values(by=feature_format.coupon_id,inplace=True)
    sort_before.to_csv(Config.path+Config.feature_path+"Coupon_from_offline_train.csv",index=None,header=None)

class user_feature():
    def __init__(self,filename):
        self.rats=pandas.read_csv(filename,header=None)
        self.len=len(self.rats)
        self.feature=Config.user_feature_format()
        self.average=[0,0,0,0,0,0]
    def get(self,user_id):
        left=0
        right=self.len-1
        if user_id<self.rats.iloc[left][self.feature.user_id] or user_id>self.rats.iloc[right][self.feature.user_id]:
            return self.average
        while left<=right:
            mid=(left+right)/2
            if self.rats.iloc[mid][self.feature.user_id]==user_id:
                return self.rats.iloc[mid]
            else:
                if self.rats.iloc[mid][self.feature.user_id]<user_id:
                    left=mid+1
                else:
                    right=mid-1
        return self.average
#user=user_feature(Config.path+Config.feature_path+"User_from_offline_train.csv")
#print user.get(723014)

class merchant_feature():
    def __init__(self,filename):
        self.rats=pandas.read_csv(filename,header=None)
        self.len=len(self.rats)
        self.feature=Config.merchant_feature_format()
        self.average=[0,0,0,0,0,0]
    def get(self,merchant_id):
        left=0
        right=self.len-1
        if merchant_id<self.rats.iloc[left][self.feature.merchant_id] or merchant_id>self.rats.iloc[right][self.feature.merchant_id]:
            return self.average
        while left<=right:
            mid=(left+right)/2
            if self.rats.iloc[mid][self.feature.merchant_id]==merchant_id:
                return self.rats.iloc[mid]
            else:
                if self.rats.iloc[mid][self.feature.merchant_id]<merchant_id:
                    left=mid+1
                else:
                    right=mid-1
        return self.average
#merchant=user_feature(Config.path+Config.feature_path+"Merchant_from_offline_train.csv")
#print merchant.get(0)

class coupon_feature():
    def __init__(self,filename):
        self.rats=pandas.read_csv(filename,header=None)
        self.len=len(self.rats)
        self.feature=Config.coupon_feature_format()
        self.average=[0,0,0,0,0,0]
        print type(self.rats.iloc[0][self.feature.coupon_id])
    def get(self,id):
        coupon_id=int(id)
        left=0
        right=self.len-1
        if coupon_id<self.rats.iloc[left][self.feature.coupon_id] or coupon_id>self.rats.iloc[right][self.feature.coupon_id]:
            return self.average
        while left<=right:
            mid=(left+right)/2
            if self.rats.iloc[mid][self.feature.coupon_id]==coupon_id:
                return self.rats.iloc[mid]
            else:
                if self.rats.iloc[mid][self.feature.coupon_id]<coupon_id:
                    left=mid+1
                else:
                    right=mid-1
        return self.average
#coupon=coupon_feature(Config.path+Config.feature_path+"Coupon_from_offline_train.csv")
#print coupon.get(10001)