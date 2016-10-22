discount_type=1
fullCut_type=0
class file_offline_train_line():
    def __init__(self):
        self.User_id = 0
        self.Merchant_id = 1
        self.Coupon_id   = 2
        self.Discount_rate = 3
        self.Distance = 4
        self.Date_received = 5
        self.Date = 6
        self.len=7
class file_online_train_line():
    def __init__(self):
        self.dic={}
        self.dic["User_id"] = 0
        self.dic["Merchant_id"] = 1
        self.dic["Action"] = 2
        self.dic["Coupon_id"] = 3
        self.dic["Discount_rate"] = 4
        self.dic["Date_received"] = 5
        self.dic["Date"] = 6
class file_offline_test_line():
    def __init__(self):
        self.dic={}
        self.dic["User_id"] = 0
        self.dic["Merchant_id"] = 1
        self.dic["Coupon_id"] = 2
        self.dic["Discount_rate"] = 3
        self.dic["Distance"] = 4
        self.dic["Date_received"] = 5
class user_feature_format():
    def __init__(self):
        self.user_id=0
        self.coupon_consump=1
        self.not_coupon_consump=2
        self.total_coupon=3
        self.coupon_consump_div_total_coupon=4
        self.coupon_consump_div_total_consump=5
        self.len=6
class merchant_feature_format():
    def __init__(self):
        self.merchant_id=0
        self.coupon_consump=1
        self.not_coupon_consump=2
        self.total_coupon=3
        self.coupon_consump_div_total_coupon=4
        self.coupon_consump_div_total_consump=5
        self.len=6
class coupon_feature_format():
    def __init__(self):
        self.coupon_id=0
        self.fullCut_or_discount=1
        self.full=2
        #self.cut=3
        self.discount=3
        self.used_coupon=4
        self.not_used_coupon=5
        self.use_ratio=6


path="Data/"
feature_path="Feature/"