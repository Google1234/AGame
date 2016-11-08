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
    def __init__(self,filename,recalculate_avr):
        self.rats=pandas.read_csv(filename,header=None)
        self.len=len(self.rats)
        self.feature=Config.user_feature_format()
        self.average = [0.000000,0.139742,1.300617,1.952554,0.073788,0.337112]
        if recalculate_avr==True:
            self.average=self.rats.mean(axis=0)
            self.average[self.feature.user_id]=0
        #print self.average
    def get(self,user_id):
        left=0
        right=self.len-1
        if user_id<self.rats.iloc[left][self.feature.user_id] or user_id>self.rats.iloc[right][self.feature.user_id]:
            return self.average
        while left<=right:
            mid=(left+right)/2
            if self.rats.iloc[mid][self.feature.user_id]==user_id:
                return [i for i in self.rats.iloc[mid]]
            else:
                if self.rats.iloc[mid][self.feature.user_id]<user_id:
                    left=mid+1
                else:
                    right=mid-1
        return self.average
#user=user_feature(Config.path+Config.feature_path+"User_from_offline_train.csv",True)
#print user.get(723014)

class merchant_feature():
    def __init__(self,filename,recalculate_avr):
        self.rats=pandas.read_csv(filename,header=None)
        self.len=len(self.rats)
        self.feature=Config.merchant_feature_format()
        self.average=[0.000000,8.958051,83.375163,125.167201,0.295601,0.089236]
        if recalculate_avr==True:
            self.average=self.rats.mean(axis=0)
            self.average[self.feature.merchant_id]=0
        #print self.average
    def get(self,merchant_id):
        left=0
        right=self.len-1
        if merchant_id<self.rats.iloc[left][self.feature.merchant_id] or merchant_id>self.rats.iloc[right][self.feature.merchant_id] :
            return self.average
        while left<=right:
            mid=(left+right)/2
            if self.rats.iloc[mid][self.feature.merchant_id]==merchant_id:
                return [i for i in self.rats.iloc[mid]]
            else:
                if self.rats.iloc[mid][self.feature.merchant_id]<merchant_id:
                    left=mid+1
                else:
                    right=mid-1
        return self.average
#merchant=user_feature(Config.path+Config.feature_path+"Merchant_from_offline_train.csv",True)
#print merchant.get(0)

class coupon_feature():
    def __init__(self,filename,recalculate_avr):
        self.rats=pandas.read_csv(filename,header=None)
        self.len=len(self.rats)
        self.feature=Config.coupon_feature_format()
        self.average=[0.000000,0.898039,40.118595,0.208894,7.740220,100.410720,0.197088]
        if recalculate_avr == True:
            self.average = self.rats.mean(axis=0)
            self.average[self.feature.coupon_id] = 0
            print self.average
    def get(self,id):
        if id=="null":
            return self.average
        coupon_id=int(id)
        left=0
        right=self.len-1
        if coupon_id<self.rats.iloc[left][self.feature.coupon_id] or coupon_id>self.rats.iloc[right][self.feature.coupon_id]:
            return self.average
        while left<=right:
            mid=(left+right)/2
            if self.rats.iloc[mid][self.feature.coupon_id]==coupon_id:
                return [i for i in self.rats.iloc[mid]]
            else:
                if self.rats.iloc[mid][self.feature.coupon_id]<coupon_id:
                    left=mid+1
                else:
                    right=mid-1
        return self.average
class distance_feature():
    def __init__(self,filename):
        self.rats=pandas.read_csv(filename,header=None)
        self.len=len(self.rats)
        self.feature=Config.file_offline_train_line()
    def get(self,user_id,merchant_id):
        if id=="null" or merchant_id=="null":
            return [self.average]
        left=0
        right=self.len-1
        if user_id<self.rats.iloc[left][self.feature.User_id] or user_id>self.rats.iloc[right][self.feature.User_id]:
            return [self.average]
        while left<=right:
            mid=(left+right)/2
            if self.rats.iloc[mid][self.feature.User_id]==user_id:
                up=mid
                while self.rats.iloc[up][self.feature.User_id]==user_id:
                    if self.rats.iloc[up][self.feature.Merchant_id]==merchant_id:
                        return [int(self.rats.iloc[up][self.feature.Distance])]
                    else:
                        up-=1
                down=mid
                while self.rats.iloc[down][self.feature.User_id]==user_id:
                    if self.rats.iloc[down][self.feature.Merchant_id]==merchant_id:
                        return [int(self.rats.iloc[down][self.feature.Distance])]
                    else:
                        down+=1
                return [Config.avr_distance]
            else:
                if self.rats.iloc[mid][self.feature.User_id]<user_id:
                    left=mid+1
                else:
                    right=mid-1
        return [self.average]

class user_Collaborate_merchant():
    def __init__(self):
        return 0
class user_Collaborate_coupon():
    def __init__(self):
        return 0
class merchant_Collaborate_merchant():
    def __init__(self):
        return 0

