import pandas
import Config

class collaborate_feature_format():
    def __init__(self):
        self.user_id=0
        self.coupon_id=1
        #user collaborate coupon
        self.coupon_id_be_used=2
        self.coupon_id_not_be_used=3
        self.coupon_id_total=4
        self.coupon_id_be_used_div_coupon_id_total=5

        self.len=6
def bulid_collaborate_feature():
    #collaborate_feature from offline train file
    rats=pandas.read_csv(Config.path+"ccf_offline_stage1_train._sorted_by_User_id.csv",header=None)
    feature_format=collaborate_feature_format()
    file_format=Config.file_offline_train_line()
    result=[]
    line2=0
    coupon_dic={}
    last_user=rats.iloc[0][file_format.User_id]
    print len(rats)/10000
    for line in xrange(len(rats)):
        user=rats.iloc[line][file_format.User_id]
        coupon=rats.iloc[line][file_format.Coupon_id]
        if user!=last_user:
            for key in coupon_dic.keys():
                t=coupon_dic[key]
                result[t][feature_format.coupon_id_be_used_div_coupon_id_total] = \
                        result[t][feature_format.coupon_id_be_used] * 1.0 / result[t][
                            feature_format.coupon_id_total]
            last_user=user
            coupon_dic.clear()

        if coupon!="null":
            if coupon not in coupon_dic.keys():
                result.append([user,coupon]+[0 for i in range(feature_format.len-2)])
                coupon_dic[coupon]=line2
                line2 += 1
            t=coupon_dic[coupon]
            result[t][feature_format.coupon_id_total]+=1
            if rats.iloc[line][file_format.Date]!="null":
                result[t][feature_format.coupon_id_be_used]+=1
            else:
                result[t][feature_format.coupon_id_not_be_used]+=1
        if line%10000==0:
            print line/10000

    for key in coupon_dic.keys():
        t = coupon_dic[key]
        result[t][feature_format.coupon_id_be_used_div_coupon_id_total] = \
            result[t][feature_format.coupon_id_be_used] * 1.0 / result[t][
                feature_format.coupon_id_total]
    del coupon_dic

    #write to file
    pandas.DataFrame(result).to_csv(Config.feature_path+"User_collaborate_coupon_from_offline_train.csv",index=None,header=None)
    ####

class collaborate_feature():
    def __init__(self,filename,recalculate_avr=True):
        self.rats=pandas.read_csv(filename,header=None)
        self.len=len(self.rats)
        self.feature=collaborate_feature_format()
        self.average = [-1.0, -1.0, 0.078706729674928325, 1.0210303646641428, 1.099737094339071, 0.054013079393102216]
        if recalculate_avr==True:
            a=self.rats.mean(axis=0)
            a[self.feature.user_id]=-1
            a[self.feature.coupon_id]=-1
            for i in range(len(a)):
                self.average[i]=a[i]
        print ("average:",[i for i in self.average])
    def get(self,user_id,coupon_id):
        left=0
        right=self.len-1
        if user_id<self.rats.iloc[left][self.feature.user_id] or user_id>self.rats.iloc[right][self.feature.user_id]:
            return self.average #no user matched ,return all user all coupon average
        while left<=right:
            mid=(left+right)/2
            if self.rats.iloc[mid][self.feature.user_id]==user_id:
                up=mid
                sum=[0 for i in range(self.feature.len)]
                count=0
                while up>=0 and self.rats.iloc[up][self.feature.user_id]==user_id:
                    if self.rats.iloc[up][self.feature.coupon_id]==coupon_id:
                        return [fea for fea in self.rats.iloc[up]]
                    else:
                        for i in range(len(sum)):
                            sum[i]+=self.rats.iloc[up][i]
                        count+=1
                    up-=1
                down=mid+1
                while down<len(self.rats) and self.rats.iloc[down][self.feature.user_id]==user_id:
                    if self.rats.iloc[down][self.feature.coupon_id]==coupon_id:
                        return [fea for fea in self.rats.iloc[down]]
                    else:
                        for i in range(len(sum)):
                            sum[i]+=self.rats.iloc[down][i]
                        count+=1
                    down+=1
                for i in range(len(sum)):
                    sum[i]=sum[i]*1.0/count
                sum[self.feature.user_id]=user_id
                sum[self.feature.coupon_id]=-1
                return sum #match user but not match merchant ,return user all coupon average
            else:
                if self.rats.iloc[mid][self.feature.user_id]<user_id:
                    left=mid+1
                else:
                    right=mid-1
        return self.average#no user matched ,return all user all merchant average

#from file bulid collaborate feature to file Data/feature/User_collaborate_coupon_from_offline_train.csv
#bulid_collaborate_feature()
#

#get (user,coupon) feature from file Data/feature/User_collaborate_coupon_from_offline_train.csv
#collaborate_feature(Config.feature_path+"User_collaborate_coupon_from_offline_train.csv",True)