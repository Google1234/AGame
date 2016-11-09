import pandas
import Config

class collaborate_feature_format():
    def __init__(self):
        self.user_id=0
        self.merchant_id=1
        #user collaborate merchant
        self.user_use_coupon_consume_in_merchant=2
        self.user_not_use_coupon_consume_in_merchante=3
        self.user_total_consump_in_merchant=4
        self.user_use_coupon_in_merchant_div_user_total_consump_in_merchant=5
        self.user_get_coupon_in_merchant=6
        self.user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant=7
        self.len=8
def bulid_collaborate_feature():
    #collaborate_feature from offline train file
    rats=pandas.read_csv(Config.path+"ccf_offline_stage1_train._sorted_by_User_id.csv",header=None)
    feature_format=collaborate_feature_format()
    file_format=Config.file_offline_train_line()
    result=[]
    line2=0
    merchant_dic={}
    last_user=rats.iloc[0][file_format.User_id]
    print len(rats)/10000
    for line in xrange(len(rats)):
        user=rats.iloc[line][file_format.User_id]
        merchant=rats.iloc[line][file_format.Merchant_id]
        if user!=last_user:
            sum_user_use_coupon_in_merchant_div_user_total_consump_in_merchant=0
            count_user_use_coupon_in_merchant_div_user_total_consump_in_merchant=0
            t_user_use_coupon_in_merchant_div_user_total_consump_in_merchant=[]

            sum_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant=0
            count_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant=0
            t_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant=[]


            for key in merchant_dic.keys():
                t=merchant_dic[key]
                if result[t][feature_format.user_total_consump_in_merchant]!=0:
                    result[t][feature_format.user_use_coupon_in_merchant_div_user_total_consump_in_merchant]= \
                        result[t][feature_format.user_use_coupon_consume_in_merchant]*1.0/result[t][feature_format.user_total_consump_in_merchant]
                    sum_user_use_coupon_in_merchant_div_user_total_consump_in_merchant+=\
                        result[t][feature_format.user_use_coupon_in_merchant_div_user_total_consump_in_merchant]
                    count_user_use_coupon_in_merchant_div_user_total_consump_in_merchant+=1
                else:
                    t_user_use_coupon_in_merchant_div_user_total_consump_in_merchant.append(t)

                if result[t][feature_format.user_get_coupon_in_merchant]!=0:
                    result[t][feature_format.user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant] = \
                        result[t][feature_format.user_use_coupon_consume_in_merchant] * 1.0 / result[t][
                            feature_format.user_get_coupon_in_merchant]
                    sum_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant+=\
                        result[t][feature_format.user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant]
                    count_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant+=1
                else:
                    t_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant.append(t)


            if count_user_use_coupon_in_merchant_div_user_total_consump_in_merchant!=0:
                avr=sum_user_use_coupon_in_merchant_div_user_total_consump_in_merchant*1.0\
                    /count_user_use_coupon_in_merchant_div_user_total_consump_in_merchant
            else:
                avr=-1

            for t1 in t_user_use_coupon_in_merchant_div_user_total_consump_in_merchant:
                result[t1][feature_format.user_use_coupon_in_merchant_div_user_total_consump_in_merchant] =avr

            if count_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant!=0:
                avr=sum_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant*1.0\
                    /count_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant
            else:
                avr=-1
            for t2 in t_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant:
                result[t2][feature_format.user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant]=avr

            last_user=user
            merchant_dic.clear()

        if merchant not in merchant_dic.keys():
            result.append([user,merchant]+[0 for i in range(feature_format.len-2)])
            merchant_dic[merchant]=line2
            line2 += 1
        t=merchant_dic[merchant]
        if rats.iloc[line][file_format.Date]=="null":
            if rats.iloc[line][file_format.Coupon_id]!="null":
                result[t][feature_format.user_get_coupon_in_merchant]+=1
        else:
            result[t][feature_format.user_total_consump_in_merchant]+=1
            if rats.iloc[line][file_format.Coupon_id] != "null":
                result[t][feature_format.user_use_coupon_consume_in_merchant] += 1
                result[t][feature_format.user_get_coupon_in_merchant] += 1
            else:
                result[t][feature_format.user_not_use_coupon_consume_in_merchante] += 1

        if line%10000==0:
            print line/10000

    sum_user_use_coupon_in_merchant_div_user_total_consump_in_merchant = 0
    count_user_use_coupon_in_merchant_div_user_total_consump_in_merchant = 0
    t_user_use_coupon_in_merchant_div_user_total_consump_in_merchant = []

    sum_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant = 0
    count_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant = 0
    t_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant = []

    for key in merchant_dic.keys():
        t = merchant_dic[key]
        if result[t][feature_format.user_total_consump_in_merchant] != 0:
            result[t][feature_format.user_use_coupon_in_merchant_div_user_total_consump_in_merchant] = \
                result[t][feature_format.user_use_coupon_consume_in_merchant] * 1.0 / result[t][
                    feature_format.user_total_consump_in_merchant]
            sum_user_use_coupon_in_merchant_div_user_total_consump_in_merchant += \
                result[t][feature_format.user_use_coupon_in_merchant_div_user_total_consump_in_merchant]
            count_user_use_coupon_in_merchant_div_user_total_consump_in_merchant += 1
        else:
            t_user_use_coupon_in_merchant_div_user_total_consump_in_merchant.append(t)

        if result[t][feature_format.user_get_coupon_in_merchant] != 0:
            result[t][feature_format.user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant] = \
                result[t][feature_format.user_use_coupon_consume_in_merchant] * 1.0 / result[t][
                    feature_format.user_get_coupon_in_merchant]
            sum_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant += \
                result[t][feature_format.user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant]
            count_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant += 1
        else:
            t_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant.append(t)

    if count_user_use_coupon_in_merchant_div_user_total_consump_in_merchant != 0:
        avr = sum_user_use_coupon_in_merchant_div_user_total_consump_in_merchant * 1.0 \
              / count_user_use_coupon_in_merchant_div_user_total_consump_in_merchant
    else:
        avr = -1

    for t1 in t_user_use_coupon_in_merchant_div_user_total_consump_in_merchant:
        result[t1][feature_format.user_use_coupon_in_merchant_div_user_total_consump_in_merchant] = avr

    if count_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant != 0:
        avr = sum_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant * 1.0 \
              / count_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant
    else:
        avr = -1
    for t2 in t_user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant:
        result[t2][feature_format.user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant] = avr

    del merchant_dic

    print result
    #all user all merchant average
    sum_coupon_consume_div_user_total_consump=0.0
    count_coupon_consume_div_user_total_consump=0
    sum_coupon_consume_div_user_get_coupon=0.0
    count_coupon_consume_div_user_get_coupon=0
    for line in xrange(len(result)):
        #cal self.user_use_coupon_in_merchant_div_user_total_consump_in_merchant
        if result[line][feature_format.user_use_coupon_in_merchant_div_user_total_consump_in_merchant]!=-1:
            sum_coupon_consume_div_user_total_consump+=\
                result[line][feature_format.user_use_coupon_in_merchant_div_user_total_consump_in_merchant]
            count_coupon_consume_div_user_total_consump+=1
        #cal user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant
        if result[line][feature_format.user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant]!=-1:
            sum_coupon_consume_div_user_get_coupon+=\
                result[line][feature_format.user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant]
            count_coupon_consume_div_user_get_coupon+=1
    avr_coupon_consume_div_user_total_consump=sum_coupon_consume_div_user_total_consump/count_coupon_consume_div_user_total_consump
    avr_coupon_consume_div_user_get_coupon=sum_coupon_consume_div_user_get_coupon/count_coupon_consume_div_user_get_coupon
    print avr_coupon_consume_div_user_total_consump,avr_coupon_consume_div_user_get_coupon
    #use avr to fill feature=-1
    for line in xrange(len(result)):
        if result[line][feature_format.user_use_coupon_in_merchant_div_user_total_consump_in_merchant] == -1:
            result[line][feature_format.user_use_coupon_in_merchant_div_user_total_consump_in_merchant]=avr_coupon_consume_div_user_total_consump
        if result[line][feature_format.user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant]==-1:
            result[line][feature_format.user_use_coupon_consume_in_merchant_div_user_get_coupon_in_merchant]=avr_coupon_consume_div_user_get_coupon
    #
    #write to file
    pandas.DataFrame(result).to_csv(Config.feature_path+"User_collaborate_merchant_from_offline_train.csv",index=None,header=None)
    ####


class collaborate_feature():
    def __init__(self,filename,recalculate_avr=True):
        self.rats=pandas.read_csv(filename,header=None)
        self.len=len(self.rats)
        self.feature=collaborate_feature_format()
        self.average = [-1.0, -1.0, 0.081835018552976396, 0.7616621035101927, 0.84349712206316918, 0.11029295896870774, 1.1434474013891391, 0.047741463736399314]
        if recalculate_avr==True:
            a=self.rats.mean(axis=0)
            a[self.feature.user_id]=-1
            a[self.feature.merchant_id]=-1
            for i in range(len(a)):
                self.average[i]=a[i]
        print ("average:",[i for i in self.average])
    def get(self,user_id,merchant_id):
        left=0
        right=self.len-1
        if user_id<self.rats.iloc[left][self.feature.user_id] or user_id>self.rats.iloc[right][self.feature.user_id]:
            return self.average #no user matched ,return all user all merchant average
        while left<=right:
            mid=(left+right)/2
            if self.rats.iloc[mid][self.feature.user_id]==user_id:
                up=mid
                sum=[0 for i in range(self.feature.len)]
                count=0
                while up>=0 and self.rats.iloc[up][self.feature.user_id]==user_id:
                    if self.rats.iloc[up][self.feature.merchant_id]==merchant_id:
                        return [fea for fea in self.rats.iloc[up]]
                    else:
                        for i in range(len(sum)):
                            sum[i]+=self.rats.iloc[up][i]
                        count+=1
                    up-=1
                down=mid+1
                while down<len(self.rats) and self.rats.iloc[down][self.feature.user_id]==user_id:
                    if self.rats.iloc[down][self.feature.merchant_id]==merchant_id:
                        return [fea for fea in self.rats.iloc[down]]
                    else:
                        for i in range(len(sum)):
                            sum[i]+=self.rats.iloc[down][i]
                        count+=1
                    down+=1
                for i in range(len(sum)):
                    sum[i]=sum[i]*1.0/count
                sum[self.feature.user_id]=user_id
                sum[self.feature.merchant_id]=-1
                return sum #match user but not match merchant ,return user all merchant average
            else:
                if self.rats.iloc[mid][self.feature.user_id]<user_id:
                    left=mid+1
                else:
                    right=mid-1
        return self.average#no user matched ,return all user all merchant average

#from file bulid collaborate feature to file Data/feature/User_collaborate_merchant_from_offline_train.csv
#bulid_collaborate_feature()
#
#get (user,merchant) feature from file Data/feature/User_collaborate_merchant_from_offline_train.csv
#collaborate_feature(Config.feature_path+"User_collaborate_merchant_from_offline_train.csv",True)
#

